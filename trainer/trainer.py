import torch
import os
from tqdm import tqdm

from .utils import build_model, get_dataloader
from .scheduler import linear_warmup_decay_scheduler, cosine_warmup_scheduler
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group


class Trainer:
    def __init__(self, model_args : dict, train_args : dict):
        self.model_args = model_args
        self.train_args = train_args
        
        self.is_float16 = train_args.get('float16', False)
        self.train_type = train_args['train_type']
        self.mix_precision = train_args.get('mix_precision', False) and not self.is_float16
        self.mix_precision = self.mix_precision and not train_args.get('accelerate', False)
        
        self.device = train_args['device']
        
        self.model = build_model(model_args)
        if self.is_float16:
            self.model.half()
        
        self.train_projection = model_args.get('force_text_projection', False)
        self.text_projection_iters = train_args.get('text_projection_iters', 1000)
        if self.train_projection:
            self.model.setup_training(train_text=False, device=self.device)
        else:
            self.model.setup_training(device=self.device)
        
        self.model_name = self.train_type + "_" + model_args['model_type'] + '_' + model_args['text_model'] + '_' + model_args['vision_model']
        self.model_name = self.model_name.replace('/','-')
            
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.train_args['lr'], weight_decay = self.train_args['weight_decay'],betas=(0.9, self.train_args['beta2']))
        
        if self.train_type == 'ddp':
            torch.cuda.set_device(self.device)  # master gpu takes up extra memory
            torch.cuda.empty_cache()
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, 
                                                                   device_ids = [self.device], 
                                                                   find_unused_parameters=True,
                                                                   mixed_precision=self.mix_precision)
           
        elif self.train_type == 'dp':
                self.model = torch.nn.DataParallel(self.model).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.train_args['lr'], weight_decay = self.train_args['weight_decay'],betas=(0.9, self.train_args['beta2']))
        self.epochs = self.train_args['epochs']
        self.batch_size = self.train_args['batch_size']
        
        self.dataloaders, self.samplers = get_dataloader(train_args, model_args) 
        self.len_dataloader = sum([len(loader) for loader in self.dataloaders])

        self.save_dir = self.train_args['save_dir']
        self.evaluate_every = self.train_args['evaluate_every']
        
        if self.train_args['scheduler'] == 'linear':
            self.scheduler = linear_warmup_decay_scheduler(self.optimizer, 
                                                            self.train_args['warmup_steps'], 
                                                            self.len_dataloader * self.epochs, 
                                                            self.train_args['intial_lr'], 
                                                            self.train_args['peak_lr'])
        
        elif self.train_args['scheduler'] == 'cosine':
            self.scheduler = cosine_warmup_scheduler(self.optimizer, 
                                                    self.train_args['warmup_steps'], 
                                                    self.len_dataloader * self.epochs, 
                                                    self.train_args['intial_lr'])
        
        if self.mix_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            

            
    def load_checkpoint(self, checkpoint):
        self.model.load_checkpoint(checkpoint['model_state_dict'])
        
    def distributed_update(self, sampler, epoch):
        if self.train_type == 'ddp':
            sampler.set_epoch(epoch)
        
    def _mini_batch_train(self, images=None, texts_1=None, texts_2=None):
        self.optimizer.zero_grad()  
        if self.mix_precision:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss = self._forward_pass(images, texts_1, texts_2)
                # loss = loss.sum() # sum() to make it a scalar
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = self._forward_pass(images, texts_1, texts_2)
            # loss = loss.sum()
            loss.sum().backward()
            self.optimizer.step()
        return loss
                
    def _forward_pass(self, images=None, texts_1=None, texts_2=None):
        if texts_2 is None:
            return self.model(images, texts_1)
        if images is None:
            return self.model(texts_1, texts_2)
        return self.model(images, texts_1, texts_2)
    
    def _unfreeze_text(self):
        self.train_projection = False
        pass
    
        
    def train(self):
        # Not reporting the loss on WanDB
        self.model.train()
        losses = []
        i = 0
        min_loss = 1e9
        for epoch in range(self.epochs):
            for dataloader, sampler in zip(self.dataloaders, self.samplers):
                self.distributed_update(sampler, epoch)
                for images, texts in tqdm(dataloader, desc = f'Epoch {epoch + 1}'):
                    i+=1
                    loss = self._mini_batch_train(images = images, texts_1=texts)
                    self.scheduler.step()
                    if i % self.evaluate_every == 0:
                        self.check_save_model(loss, min_loss)
                    
                    if self.train_projection and i > self.text_projection_iters:
                        self._unfreeze_text()

                    losses.append(loss.item())
        return losses
    
    def save_checkpoint(self):
        if not self.model.train_vision:
            if self.train_projection:
                self.model.save_text_checkpoint(os.path.join(self.save_dir, f'text_{self.model_name}'))
            else:
                self.model.save_projection_checkpoint(os.path.join(self.save_dir, f'text_{self.model_name}'))
        else:
            self.model.save_checkpoint(os.path.join(self.save_dir, f'{self.model_name}.pth'))
    
    def check_save_model(self, loss, min_loss):
        if loss.item() < min_loss:
            if self.train_type == 'ddp':
                if self.gpu_id == 0:
                    self.save_checkpoint()
            else:
                self.save_checkpoint()
                
        return min(loss.sum().item(), min_loss)  
    
class CrossLingualTrainer(Trainer):
    def __init__(self, model_args, train_args):
        super(CrossLingualTrainer, self).__init__(model_args, train_args)
        
    def train(self):
        self.model.train()
        losses = []
        min_loss = 1e9
        i = 0
        for epoch in range(self.epochs):
            for dataloader, sampler in zip(self.dataloaders, self.samplers):
                self.distributed_update(sampler, epoch)
                for texts_1, texts_2 in tqdm(dataloader, desc = f'Epoch {epoch + 1}'):
                    i += 1
                    loss = self._mini_batch_train(texts_1 = texts_1, texts_2 = texts_2)
                    self.scheduler.step()
                    
                    if i % self.evaluate_every == 0:
                        min_loss = self.check_save_model(loss, min_loss) 
                            
                    losses.append(loss.sum().item())
                    
                    
        return losses
    
class mCLIPTrainer(Trainer):
    def __init__(self, model_args, train_args):
        super(mCLIPTrainer, self).__init__(model_args, train_args)
        
    def train(self):
        self.model.train()
        losses = []
        min_loss = 1e9
        i = 0
        for epoch in range(self.epochs):
            for dataloader, sampler in zip(self.dataloaders, self.samplers):
                self.distributed_update(sampler, epoch)
                for images, texts_1, texts_2 in tqdm(dataloader, desc = f'Epoch {epoch + 1}'):
                    i += 1
                    loss = self._mini_batch_train(images = images, texts_1 = texts_1, texts_2 = texts_2)
                    self.scheduler.step()
                    
                    if i % self.evaluate_every == 0:
                        min_loss = self.check_save_model(loss, min_loss) 
                            
                    losses.append(loss.sum().item())
                    
        return losses

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
        if model_args['model_type'] == 'crosslingual':
            trainer = CrossLingualTrainer(model_args, train_args)
        elif model_args['model_type'] == 'mclip':
            trainer = mCLIPTrainer(model_args, train_args)
        else:
            trainer = Trainer(model_args, train_args)
            
        losses = trainer.train()

        destroy_process_group()
        return losses
    
    
def ddp_train(train_args: dict, model_args: dict):

    # Each process control a single gpu
    if train_args.get('accelerate') == True:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    world_size = torch.cuda.device_count()
    mp.spawn(main_ddp, args=(world_size, train_args, model_args), nprocs=world_size)