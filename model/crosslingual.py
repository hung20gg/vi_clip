import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer,  AutoProcessor, AutoModel, CLIPModel, SiglipModel
from .utils import mean_pooling, count_parameters, open_image, all_gather_default, CLIPImage, CLIPText
from .lossfn import sigliploss, cliploss
from .model import TextEncoder
from collections.abc import Iterable 

import gc


class CrossLingual2:
    def __init__(self, text_model, clip_model, device = None, max_length = 64, **kwargs):

        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
        model = AutoModel.from_pretrained(clip_model)
        if hasattr(model, 'projection_dim'):
            self.projection_dim = model.projection_dim
        else:
            self.projection_dim = 768      
            
        self.clip_text_model = CLIPText(model)  
        self.clip_tokenizer = AutoTokenizer.from_pretrained(clip_model, use_fast=True)
        
        del model
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except:
            print('No GPU available')
            
        self.text_model = TextEncoder(text_model, max_length, 'mse', projection_dim = self.projection_dim , device=self.device)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model, use_fast=True)
        self.encode_text = self.text_model.encode_text
        
    def encode_clip_text(self, text):
        inputs = self.clip_tokenizer(text=text, max_length = self.text_model.max_length, padding=True, truncation=True, return_tensors='pt').to(self.device)

        self.clip_text_model.eval()
        outputs = self.clip_text_model(**inputs)
                
        emb_norm = torch.norm(outputs, dim=1, keepdim=True)
        return outputs / (emb_norm + 1e-8)
        
class CrossLingual(nn.Module):
    """
    
        Training BERT-based text model using CLIP/SigLIP text tower, using MSE loss.
        The implementation is similar to the CLIP class in the model.py file.

    """
    def __init__(self,
                 clip_model = 'google/siglip-base-patch16-224',
                 text_model = 'vinai/phobert-base-v2',
                 load_vision = False,
                 max_length = 64,
                 **kwargs):
        super(CrossLingual, self).__init__()
        
        self.loss_fn = nn.MSELoss()

        print('Loading CLIP/SigLIP model')
        self.vision = load_vision
        
        model = AutoModel.from_pretrained(clip_model)
        
        if hasattr(model, 'projection_dim'):
            self.projection_dim = model.projection_dim
        else:
            self.projection_dim = 768
            
        self.processor = AutoProcessor.from_pretrained(clip_model, is_training=False)
        if load_vision:
            self.vision_model = CLIPImage(model)  
            print(f'Number of vision model parameters: {count_parameters(self.vision_model)}')
        
        self.clip_text_model = CLIPText(model)
        
        del model
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except:
            print('No GPU available')
        
        self.text_model = AutoModel.from_pretrained(text_model)
        if self.projection_dim != self.text_model.config.hidden_size:
            self.text_projection = nn.Linear(self.text_model.config.hidden_size, self.projection_dim)
        else:
            self.text_projection = nn.Identity()
        self.tokenizer = AutoTokenizer.from_pretrained(text_model, use_fast=True)
        print(f'Number of text model parameters: {count_parameters(self.text_model)}')
        
        self.train_vision = False
        self.train_clip_text = False
        self.train_text = False
        
        self.max_length = max_length
        
    def setup_device(self, device = None):
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def setup_training(self, train_vision = False, train_clip_text = False, train_text = True, device = None):
        self.setup_device(device)
        
        self.train_vision = train_vision & self.vision
        self.train_clip_text = train_clip_text
        self.train_text = train_text
        
        if self.vision and not self.train_vision:
            print('Freezing vision model')
            for param in self.vision_model.parameters():
                param.requires_grad = False

        if not self.train_clip_text:
            print('Freezing CLIP text model')
            for param in self.clip_text_model.parameters():
                param.requires_grad = False
                
        num_params = count_parameters(self)
        print(f'Number of parameters: {num_params}')
        
    def load_checkpoint(self, checkpoint):
        self.load_state_dict(torch.load(checkpoint))
        
    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)
        
    def load_text_checkpoint(self, checkpoint):
        self.text_model.load_state_dict(torch.load(checkpoint))
        
    def save_text_checkpoint(self, path, using_automodel = True):
        if not using_automodel:
            torch.save(self.text_model.state_dict(), path)
        else:
            self.text_model.save_pretrained(path)
        
    def transform_image(self, images):
        assert self.vision, 'Vision model is not loaded'
        
        if isinstance(images, torch.Tensor):
            return images.to(self.device)
        if isinstance(images, Iterable) and all(isinstance(i, str) for i in images):
            images = np.array([np.array(open_image(i)) for i in images])
        
        return self.processor(images, return_tensors='pt').to(self.device)

    def encode_image(self, image):
        assert self.vision, 'Vision model is not loaded'
        
        image = self.transform_image(image)
        
        if not self.train_vision:
            self.vision_model.eval()
        output = self.vision_model(**image)

        emb_norm = torch.norm(output, dim=1, keepdim=True)
        return output / (emb_norm + 1e-8)
    
    def encode_text(self, text, result = 'mean'):
        inputs = self.tokenizer(text, max_length=self.max_length, padding=True, truncation=True, return_tensors='pt').to(self.device)
    
        outputs = self.text_model(**inputs).last_hidden_state
                    
        if result == 'eos':
            emb_text = outputs
            emb_text = emb_text[torch.arange(emb_text.shape[0]), torch.where(inputs['input_ids'] == self.tokenizer.eos_token_id)[1]]
        
        else:
            emb_text = mean_pooling(outputs, inputs['attention_mask'])
        
        emb_text = self.text_projection(emb_text)
        emb_norm = torch.norm(emb_text, dim=1, keepdim=True)
        return emb_text / (emb_norm + 1e-8)
    
    def encode_clip_text(self, text):
        inputs = self.processor(text=text, max_length = self.max_length, padding=True, truncation=True, return_tensors='pt').to(self.device)

        if not self.train_clip_text:
            self.clip_text_model.eval()
        outputs = self.clip_text_model(**inputs)
                
        emb_norm = torch.norm(outputs, dim=1, keepdim=True)
        return outputs / (emb_norm + 1e-8)
    
    def encode(self, images, texts):
        image_embed = self.encode_image(images)
        text_embed = self.encode_text(texts)
        return image_embed, text_embed
        
    def forward(self, text_1, text_2, train_type = 'single', **kwargs):
        
        y_pred = self.encode_text(text_1)
        y_true = self.encode_clip_text(text_2)
        
        return self.loss_fn(y_pred, y_true)
    
class mCLIP(CrossLingual):
    """
        mCLIP model implementation.
        mCLIP is a model that uses two different text embeddings to calculate the loss.
        - IT loss: Image to New Text loss (CLIP loss)
        - TT loss: Text to Text loss (CLIP loss)
        
        Loss = IT_loss + lambda * TT_loss (usually lambda = 0.1)
    """
    def __init__(self,
                 clip_model = 'google/siglip-base-patch16-224',
                 text_model = 'vinai/phobert-base-v2',
                 load_vision = True,
                 device = None,
                 lambda_ = 0.1,
                 init_scale = 10,
                 init_bias = -10):
        super(mCLIP, self).__init__(clip_model, text_model, load_vision, device)
        
        self.siglip = 'siglip' in clip_model.lower()
        if self.siglip:
            self._loss_fn = sigliploss
            self.logit_scale_tt = nn.Parameter(torch.ones(1) * torch.log(torch.ones(1)* init_scale))
            self.logit_bias_tt  = nn.Parameter(torch.ones(1) * init_bias)
            
            self.logit_scale_it = nn.Parameter(torch.ones(1) * torch.log(torch.ones(1)* init_scale))
            self.logit_bias_it  = nn.Parameter(torch.ones(1) * init_bias)
        else:
            self.logit_scale = nn.Parameter(torch.ones(1) * torch.log(torch.tensor(1/0.07)))
            self._loss_fn = cliploss
        self.lambda_ = lambda_
        
    def setup_training(self, train_vision=True, train_clip_text=False, train_text=True, device=None):
        self.setup_device(device)
        self.train_vision = train_vision & self.vision
        self.train_clip_text = train_clip_text
        self.train_text = train_text
        
        if self.vision and not self.train_vision:
            print('Freezing vision model')
            for param in self.vision_model.parameters():
                param.requires_grad = False
        if not self.train_clip_text:
            print('Freezing CLIP text model')
            for param in self.clip_text_model.parameters():
                param.requires_grad = False
                
        num_params = count_parameters(self)
        print(f'Number of parameters: {num_params}')
        
    def forward(self, image, text_1, text_2, train_type = 'single', **kwargs):
        assert self.vision, 'Vision model is not loaded'
        
        image_embed = self.encode_image(image)
        text_embed_clip = self.encode_clip_text(text_2)
        text_embed = self.encode_text(text_1)
        
        if self.siglip:
            TTloss = self._loss_fn(image_embed, text_embed_clip, self.logit_scale_tt, self.logit_bias_tt, ddp = train_type == 'ddp')
            ITloss = self._loss_fn(image_embed, text_embed, self.logit_scale_it, self.logit_bias_it, ddp = train_type == 'ddp')
        else:
            TTloss = self._loss_fn(text_embed, text_embed_clip, temperature=self.logit_scale, ddp = train_type == 'ddp')
            ITloss = self._loss_fn(image_embed, text_embed, temperature=self.logit_scale, ddp = train_type == 'ddp')
        
        return ITloss + self.lambda_ * TTloss
        
        
        
        