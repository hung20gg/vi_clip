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
        else:
            self.model = model_args
        
        self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.train_args['lr'], weight_decay = self.train_args['weight_decay'],betas=(0.9,0.95))
        self.epochs = self.train_args['epochs']

        self.batch_size = self.train_args['batch_size']
        
        self.dataloaders = get_dataloader(train_args)
        self.len_dataloader = [len(loader) for loader in self.dataloaders]
        
        if self.train_args['scheduler'] == 'linear':
            self.scheduler = linear_warmup_decay_scheduler(self.optimizer, 
                                                            self.train_args['warmup_steps'], 
                                                            self.len_dataloader * self.epochs, 
                                                            self.train_args['lr'], 
                                                            self.train_args['peak_lr'])
        
        elif self.train_args['scheduler'] == 'cosine':
            self.scheduler = cosine_warmup_scheduler(self.optimizer, 
                                                    self.train_args['warmup_steps'], 
                                                    self.len_dataloader * self.epochs, 
                                                    self.train_args['lr'])
            
    def load_checkpoint(self, checkpoint):
        self.model.load_checkpoint(checkpoint['model_state_dict'])
        
    def train(self):
        # Not reporting the loss on WanDB
        self.model.train()
        losses = []
        for epoch in range(self.epochs):
            for dataloader in self.dataloaders:
                for i, (images, texts) in enumerate(tqdm(dataloader)):
                    images = images.to(self.device)
                    texts = texts.to(self.device)
                    self.optimizer.zero_grad()
                    loss = self.model(images, texts)
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    losses.append(loss.item())
        return losses


class CrossLingualTrainer(Trainer):
    def __init__(self, model_args, train_args, device = None):
        super(CrossLingualTrainer, self).__init__(model_args, train_args, device)
        
    def train(self):
        self.model.train()
        losses = []
        for epoch in range(self.epochs):
            for dataloader in self.dataloaders:
                for i, (images, texts_1, texts_2) in enumerate(tqdm(dataloader)):
                    texts_1 = texts_1.to(self.device)
                    texts_2 = texts_2.to(self.device)
                    lang = lang.to(self.device)
                    self.optimizer.zero_grad()
                    loss = self.model(images, texts_1, texts_2)
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    losses.append(loss.item())
        return losses