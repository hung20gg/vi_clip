import torch
import os
from tqdm import tqdm

from .utils import build_model, get_dataloader
from .scheduler import linear_warmup_decay_scheduler, cosine_warmup_scheduler

class Trainer:
    def __init__(self, model_args, train_args : dict, device = None):
        self.model_args = model_args
        self.train_args = train_args
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if isinstance(self.model_args, dict):
            self.model = build_model(model_args)
            self.model_name = model_args['model_type'] + '_' + model_args['text_model'] + '_' + model_args['vision_model']
            self.model_name = self.model_name.replace('/','-')
        else:
            self.model = model_args
            self.model_name = 'custom'
        
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
        
        self.dataloaders = get_dataloader(train_args, model_args) 
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
        
    def train(self):
        # Not reporting the loss on WanDB
        self.model.setup_training()
        self.model.train()
        losses = []
        i = 0
        for epoch in range(self.epochs):
            for dataloader in self.dataloaders:
                for images, texts in tqdm(dataloader, desc = f'Epoch {epoch + 1}'):
                    i+=1
                    self.optimizer.zero_grad()
                    loss = self.model(images, texts)
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    if i % self.evaluate_every == 0:
                        if loss.item() < min_loss:
                            print(f'Loss: {loss.item()}')
                            min_loss = loss.item()
                            
                            if not self.model.train_vision:
                                self.model.save_text_checkpoint(os.path.join(self.save_dir, f'text_{self.model_name}.pth'))
                            else:
                                self.model.save_checkpoint(os.path.join(self.save_dir, f'{self.model_name}.pth'))
                     
                    losses.append(loss.item())
        return losses


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
            for dataloader in self.dataloaders:
                for texts_1, texts_2 in tqdm(dataloader, desc = f'Epoch {epoch + 1}'):
                    i += 1
                    self.optimizer.zero_grad()
                    loss = self.model(texts_1, texts_2)
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    
                    if i % self.evaluate_every == 0:
                        print(f'Loss: {loss.item()}')
                        if loss.item() < min_loss:
                            min_loss = loss.item()
                            
                            if not self.model.train_vision:
                                self.model.save_text_checkpoint(os.path.join(self.save_dir, f'text_{self.model_name}.pth'))
                            else:
                                self.model.save_checkpoint(os.path.join(self.save_dir, f'{self.model_name}.pth'))
                            
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
            for dataloader in self.dataloaders:
                for images, texts_1, texts_2 in tqdm(dataloader, desc = f'Epoch {epoch + 1}'):
                    i += 1
                    self.optimizer.zero_grad()
                    loss = self.model(images, texts_1, texts_2)
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    
                    if i % self.evaluate_every == 0:
                        if loss.item() < min_loss:
                            min_loss = loss.item()
                            
                            if not self.model.train_vision:
                                self.model.save_text_checkpoint(os.path.join(self.save_dir, f'text_{self.model_name}'))
                            else:
                                self.model.save_checkpoint(os.path.join(self.save_dir, f'{self.model_name}.pth'))
                            
                    losses.append(loss.item())
                    
        return losses