import torch
import os
from tqdm import tqdm

import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from huggingface_hub import hf_hub_download, HfApi, login
import wandb
import dotenv
dotenv.load_dotenv()


import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from .utils import build_model, get_dataloader
from .scheduler import linear_warmup_decay_scheduler, cosine_warmup_scheduler
from model import count_parameters
import threading

class Args:
    def __init__(self, world_size, rank):
        self.world_size = world_size
        self.rank = rank

class Trainer:
    def __init__(self, model_args : dict, train_args : dict):
        self.model_args = model_args
        self.train_args = train_args
         
        self.is_float16 = train_args.get('float16', False)
        self.train_type = train_args['train_type']
        self.mix_precision = train_args.get('mixed_precision', False) and not self.is_float16
        self.mix_precision = self.mix_precision and not train_args.get('accelerate', False)
        
        self.device = train_args['device']
        print(f"Device: {self.device}")
        
        self.model = build_model(model_args)
        if self.is_float16:
            self.model.half()
        
        print("Model built successfully.")
        self.train_projection = train_args.get('train_projection_only', False)
        self.text_projection_iters = train_args.get('text_projection_iters', 1000)
        if self.train_projection or self.text_projection_iters > 0:
            print("Freezing text encoder parameters...")
            self.model.setup_training(train_text=False, device=self.device)
        else:
            print("Training all text model parameters...")
            self.model.setup_training(device=self.device)
        
        self.model_name = self.train_type + "_" + model_args['model_type'] + '_' + model_args['text_model'] + '_' + model_args['vision_model']
        self.model_name = self.model_name.replace('/','-')
        
        self.wandb_report = train_args.get('wandb_project', None) is not None
        if self.wandb_report is not None:
            wandb.login(key=os.getenv("WANDB_API_KEY"))
            self.model_name = train_args['train_name'] + '_' + self.model_name
            wandb.init(name=self.model_name, project=train_args['wandb_project'])
        
        if self.train_type == 'ddp':
            torch.cuda.set_device(self.device)  # master gpu takes up extra memory
            torch.cuda.empty_cache()
            self.args = Args(world_size = train_args['world_size'], rank = self.device)
            
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, 
                                                                   device_ids = [self.device], 
                                                                   find_unused_parameters=True)
           
        elif self.train_type == 'dp':
                self.model = torch.nn.DataParallel(self.model).to(self.device)
        
        self.lr = train_args['lr']
        
        # Custom lr for text model
        if 'text' in self.model_args['model_type']:
            self.projection_lr = train_args.get('text_projection_lr', train_args['lr'])
            optimizer_params = []
            for name, param in self.model.named_parameters():
                if 'text_projection' in name:
                    optimizer_params.append({'params': param, 'lr': self.projection_lr, 'weight_decay': train_args['weight_decay'], 'betas': (0.9, train_args['beta2'])})
                
                # Text model and coef of siglip, clip
                elif param.requires_grad:
                    optimizer_params.append({'params': param, 'lr': self.lr, 'weight_decay': train_args['weight_decay'], 'betas': (0.9, train_args['beta2'])})
            self.optimizer = torch.optim.AdamW(optimizer_params)
        # Not implement custom lr for full model
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.train_args['lr'], weight_decay = self.train_args['weight_decay'],betas=(0.9, self.train_args['beta2']))
        
        self.epochs = self.train_args['epochs']
        self.batch_size = self.train_args['batch_size']
        
        self.dataloader, self.sampler = get_dataloader(train_args, model_args, device=self.device) 
        self._train_steps = len(self.dataloader) * self.epochs
                
        self.save_dir = self.train_args['save_dir']
        self.evaluate_every = self.train_args['evaluate_every']
    
        
        if self.train_args['scheduler'] == 'linear':
            self.scheduler = linear_warmup_decay_scheduler(self.optimizer, 
                                                            self.train_args['warmup_steps'], 
                                                            self._train_steps, 
                                                            self.train_args['initial_lr'], 
                                                            self.train_args['peak_lr'])
        
        elif self.train_args['scheduler'] == 'cosine':
            self.scheduler = cosine_warmup_scheduler(self.optimizer, 
                                                    self.train_args['warmup_steps'], 
                                                    self._train_steps, 
                                                    self.train_args['initial_lr'])
        
        if self.mix_precision:
            self.scaler = torch.GradScaler(self.device)
            

        
    def distributed_update(self, sampler, epoch):
        if self.train_type == 'ddp':
            sampler.set_epoch(epoch)
        
    def _mini_batch_train(self, images=None, texts_1=None, texts_2=None):
        self.optimizer.zero_grad()  
        if self.mix_precision:
            with torch.autocast(device_type=self.device, dtype=torch.float16):
                loss = self._forward_pass(images, texts_1, texts_2)
                # loss = loss.sum() # sum() to make it a scalar
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = self._forward_pass(images, texts_1, texts_2)
            if self.train_type != 'single':
                loss = loss.sum()
            loss.backward()

            self.optimizer.step()
        return loss
                
    def _forward_pass(self, images=None, texts_1=None, texts_2=None):
        if texts_2 is None:
            return self.model(images, texts_1, train_type = self.train_type)
        if images is None:
            return self.model(texts_1, texts_2, train_type = self.train_type)
        return self.model(images, texts_1, texts_2, train_type = self.train_type)
    
    def _unfreeze_text(self):
        self.train_projection = False
        for param in self.model.parameters():
            param.requires_grad = True
    
    def report_to_wandb(self, logs: dict):
        if self.wandb_report:
            wandb.log(logs)
        
    def train(self):
        
        print(f"Model name: {self.model_name}")
        print(f"Number of parameters: {count_parameters(self.model)}")
        
        self.model.train()
        losses = []
        i = 0
        min_loss = 1e9
        for epoch in range(self.epochs):
            self.distributed_update(self.sampler, epoch)
            for images, texts in tqdm(self.dataloader, desc = f'Epoch {epoch + 1}'):
                i+=1
                if not self.train_projection and i > self.text_projection_iters:
                    self._unfreeze_text()
                
                bs = images.shape[0]
                loss = self._mini_batch_train(images = images, texts_1=texts)
                
                self.scheduler.step()
                losses.append(loss.item())
                
                if i % self.train_args['log_every'] == 0:
                    self.report_to_wandb({'train/loss': loss.item(), 'step': i, 'lr': self.scheduler.get_last_lr()[0]})
                    print(f'Step {epoch + 1}, Loss: {loss.item():.4f}')
                
                if i % self.evaluate_every == 0:
                    print(f"Evaluating at iteration {i}...")
                    min_loss = self.check_save_model(loss, min_loss, bs, step=i)
                    # Push to HF in a separate thread to avoid blocking training
                    threading.Thread(target=self.push_to_hf, kwargs={'step': i}, daemon=True).start()

        return losses
    
    def save_checkpoint(self, step=None):
        if os.path.exists(self.save_dir) == False:
            os.makedirs(self.save_dir)
        self.model.save_checkpoint(os.path.join(self.save_dir, f'{self.model_name}_{step}.pth') if step is not None else os.path.join(self.save_dir, f'{self.model_name}.pth'))

    def ddp_save_checkpoint(self, step=None):
        if os.path.exists(self.save_dir) == False:
            os.makedirs(self.save_dir)
        if self.device == 0:
            model_path = os.path.join(self.save_dir, f'{self.model_name}_{step}.pth') if step is not None else os.path.join(self.save_dir, f'{self.model_name}.pth')
            ckpt = self.model.module.state_dict()
            torch.save(ckpt, model_path)

    def check_save_model(self, loss, min_loss, bs, step=None):
        if loss.item()/bs < min_loss:
            if self.train_type == 'ddp':
                if self.device == 0:
                    self.ddp_save_checkpoint(step)
            else:
                self.save_checkpoint(step)
                
        return min(loss.item(), min_loss)  
    
    def push_to_hf(self, step=None):
        api = HfApi()
        api.upload_file(
            path_or_fileobj=os.path.join(self.save_dir, f'{self.model_name}_{step}.pth') if step is not None else os.path.join(self.save_dir, f'{self.model_name}.pth'),
            path_in_repo=f'{self.model_name}_{step}.pth' if step is not None else f'{self.model_name}.pth',
            repo_id=self.train_args['hf_repo_name'],
            repo_type="model",
        )
        print(f"Model saved to Hugging Face at {self.train_args['hf_repo_name']}")
        

def ddp_setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "54321"  # select any idle port on your machine

    init_process_group(backend="nccl", rank=rank, world_size=world_size)
        
  
def main_ddp(
    rank: int,
    world_size: int,
    train_args: dict,
    model_args: dict,
):
        ddp_setup(rank, world_size)  # initialize ddp

        train_args['train_type'] = 'ddp'
        train_args['device'] = rank
        trainer = Trainer(model_args, train_args)
            
        losses = trainer.train()

        destroy_process_group()
        return losses
    
    
def ddp_train(train_args: dict, model_args: dict):

    # Each process control a single gpu
    if train_args.get('accelerate', False) == True:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    world_size = torch.cuda.device_count()
    print(f"World size: {world_size}")
    train_args['world_size'] = world_size
    
    mp.spawn(main_ddp, args=(world_size, train_args, model_args), nprocs=world_size)