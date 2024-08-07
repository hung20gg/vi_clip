import torch
import os
from tqdm import tqdm

from .utils import build_model, get_dataloader
from .scheduler import linear_warmup_decay_scheduler, cosine_warmup_scheduler

class Trainer:
    def __init__(self, model_args : dict, train_args : dict):
        self.model_args = model_args
        self.train_args = train_args
        
        self.train_type = train_args['train_type']
        
        if isinstance(self.model_args, dict):
            self.model = build_model(model_args)
            self.model_name = model_args['model_type'] + '_' + model_args['text_model'] + '_' + model_args['vision_model']
            self.model_name = self.model_name.replace('/','-')
        else:
            self.model = model_args
            self.model_name = 'custom'
        
        if self.train_type == 'ddp':
            self.gpu_id = train_args['gpu_id']
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids = [self.gpu_id])
        elif self.train_type == 'dp':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = torch.nn.DataParallel(self.model)
            self.model.to(self.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
        self.model.setup_training()
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.train_args['lr'], weight_decay = self.train_args['weight_decay'],betas=(0.9, self.train_args['beta2']))
        # else:
        #     parameters = list(self.model.text_model.parameters())
        #     if hasattr(self.model, 'logit_scale'):
        #         parameters.append(self.model.logit_scale.parameters())
        #     if hasattr(self.model, 'logit_bias'):
        #         parameters.append(self.model.logit_bias.parameters())
                
        #     self.optimizer = torch.optim.AdamW(parameters, lr = self.train_args['lr'], weight_decay = self.train_args['weight_decay'],betas=(0.9, self.train_args['beta2']))
        
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
            
    def load_checkpoint(self, checkpoint):
        self.model.load_checkpoint(checkpoint['model_state_dict'])
        
    def distributed_update(self, sampler, epoch):
        if self.train_type == 'ddp':
            sampler.set_epoch(epoch)
        
    def train(self):
        # Not reporting the loss on WanDB
        self.model.setup_training()
        self.model.train()
        losses = []
        i = 0
        min_loss = 1e9
        for epoch in range(self.epochs):
            for dataloader, sampler in zip(self.dataloaders, self.samplers):
                self.distributed_update(sampler, epoch)
                for images, texts in tqdm(dataloader, desc = f'Epoch {epoch + 1}'):
                    i+=1
                    self.optimizer.zero_grad()
                    loss = self.model(images, texts)
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    if i % self.evaluate_every == 0:
                        self.check_save_model(loss, min_loss)

                    losses.append(loss.item())
        return losses
    
    def save_checkpoint(self):
        if not self.model.train_vision:
            self.model.save_text_checkpoint(os.path.join(self.save_dir, f'text_{self.model_name}.pth'))
        else:
            self.model.save_checkpoint(os.path.join(self.save_dir, f'{self.model_name}.pth'))
    
    def check_save_model(self, loss, min_loss):
        if loss.item() < min_loss:
            if self.train_type == 'ddp':
                if self.gpu_id == 0:
                    self.save_checkpoint()
            else:
                self.save_checkpoint()
                
        return min(loss.item(), min_loss)  
    
class CrossLingualTrainer(Trainer):
    def __init__(self, model_args, train_args, device = None):
        super(CrossLingualTrainer, self).__init__(model_args, train_args, device)
        
    def train(self):
        self.model.setup_training()
        self.model.train()
        losses = []
        min_loss = 1e9
        i = 0
        for epoch in range(self.epochs):
            for dataloader, sampler in zip(self.dataloaders, self.samplers):
                self.distributed_update(sampler, epoch)
                for texts_1, texts_2 in tqdm(dataloader, desc = f'Epoch {epoch + 1}'):
                    i += 1
                    self.optimizer.zero_grad()
                    loss = self.model(texts_1, texts_2)
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    
                    if i % self.evaluate_every == 0:
                        min_loss = self.check_save_model(loss, min_loss) 
                            
                    losses.append(loss.item())
                    
                    
        return losses
    
class mCLIPTrainer(Trainer):
    def __init__(self, model_args, train_args, device = None):
        super(mCLIPTrainer, self).__init__(model_args, train_args, device)
        
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
                    self.optimizer.zero_grad()
                    loss = self.model(images, texts_1, texts_2)
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    
                    if i % self.evaluate_every == 0:
                        min_loss = self.check_save_model(loss, min_loss) 
                            
                    losses.append(loss.item())
                    
        return losses
    
    
    
def ddp_train(train_args, model_args):
    import torch.multiprocessing as mp
    from torch.utils.data.distributed import (
        DistributedSampler,
    )  # Distribute data across multiple gpus
    from torch.distributed import init_process_group, destroy_process_group


    # Each process control a single gpu
    def ddp_setup(rank: int, world_size: int):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "54321"  # select any idle port on your machine

        init_process_group(backend="nccl", rank=rank, world_size=world_size)
    
    def main_ddp(
    rank: int,
    world_size: int,
):
        ddp_setup(rank, world_size)  # initialize ddp

        train_args['train_type'] = 'ddp'
        if model_args['model_type'] == 'crosslingual':
            trainer = CrossLingualTrainer(model_args, train_args)
        elif model_args['model_type'] == 'mclip':
            trainer = mCLIPTrainer(model_args, train_args)
        else:
            trainer = Trainer(model_args, train_args)
            
        losses = trainer.train()

        destroy_process_group()
    
    world_size = torch.cuda.device_count()
    mp.spawn(main_ddp, args=(world_size), nprocs=world_size)