import torch
import torch.nn as nn
import numpy as np
from .lossfn import sigliploss, cliploss
from transformers import AutoTokenizer, AutoModel, AutoProcessor, CLIPImageProcessor
import timm
from .utils import mean_pooling, count_parameters, open_image, all_gather_default, print_detail, CLIPImage, CLIPText, AllGather
from collections.abc import Iterable 
import gc
import os


class TextEncoder(nn.Module):
    """
    Testing: New approach for training text encoder.
    No need to load the vision model, passing the embedding directly to the model.
    """
    def __init__(self, 
                 text_model, 
                 model_type = 'siglip', 
                 projection_dim = 768, 
                 max_length = 64, 
                 force_text_projection = True,
                 init_scale = 10, 
                 init_bias = -10,
                 **kwargs):
        super(TextEncoder, self).__init__()
  
        self.text_model = AutoModel.from_pretrained(text_model)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model, use_fast=True)
        self.max_length = max_length
        self.model_type = model_type
        self.force_text_projection = force_text_projection
        self.device = None
        
        if self.text_model.config.hidden_size != projection_dim or force_text_projection:
            self.text_projection = nn.Linear(self.text_model.config.hidden_size, projection_dim)
        else:
            self.text_projection = nn.Identity()
        
        if model_type == 'mse':
            self.loss_fn = nn.MSELoss()
            self.loss_type = 'mse'
            
        elif model_type in ['clip', 'lit', 'text_clip', 'text_lit']:
            self.loss_fn = cliploss
            self.logit_scale = nn.Parameter(torch.ones(1) * torch.log(torch.tensor(1/0.07)))
            self.loss_type = 'softmax'
            
        else:
            self.loss_fn = sigliploss
            self.logit_scale = nn.Parameter(torch.ones(1) * torch.log(torch.ones(1)* init_scale))
            self.logit_bias  = nn.Parameter(torch.ones(1) * init_bias)
            self.loss_type = 'sigmoid'
            
    def _setup_device(self, device = None):
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
            
    def setup_training(self, train_text = True, device = None, **kwargs):
        self._setup_device(device)
        self.train_text = train_text
        self.train_vision = False
        
        if not train_text:
            print('Freezing text model')
        for param in self.text_model.parameters():
            param.requires_grad = train_text
     
    # ========================   
    # Saving and loading checkpoints    
    def load_checkpoint(self, checkpoint):
        self.load_state_dict(torch.load(checkpoint))
        
    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)
        
    def load_text_checkpoint(self, checkpoint):
        self.text_model.load_state_dict(torch.load(checkpoint))
        
    def save_text_checkpoint(self, path, using_automodel = True):
        if not using_automodel:
            torch.save(self.text_model.state_dict(), os.path.join(path, 'text_model.pth'))
        else:
            self.text_model.save_pretrained(path)
        self.save_projection_checkpoint(path)
         
    def save_projection_checkpoint(self, path):
        if hasattr(self, 'text_projection'):
            torch.save(self.text_projection.state_dict(), os.path.join(path, 'text_projection.pth'))
    
    # ========================
        
    def tokenize(self, text):
        return self.tokenizer(text, max_length=self.max_length, padding=True, truncation=True, return_tensors='pt').to(self.device)
    
    def encode_text(self, text, result = 'mean'):
        
        if isinstance(text, torch.Tensor):
            emb_text = text.to(self.device)
        else:
            inputs = self.tokenize(text)
            outputs = self.text_model(**inputs).last_hidden_state
            
            if result == 'eos':
                emb_text = outputs
                emb_text = emb_text[torch.arange(emb_text.shape[0]), torch.where(inputs['input_ids'] == self.tokenizer.eos_token_id)[1]]
            else:
                emb_text = mean_pooling(outputs, inputs['attention_mask'])
            
        emb_text = self.text_projection(emb_text)
        
        emb_norm = torch.norm(emb_text, dim=1, keepdim=True)
        return emb_text / (emb_norm + 1e-8)
    
    def forward(self, images, texts, train_type = 'single', **kwargs):
        # texts: y_pred, images: y_train
        
        texts = self.encode_text(texts)
        images = torch.tensor(images).to(self.device)
                
        if self.loss_type == 'sigmoid':
            return self.loss_fn(images, texts, self.logit_scale, self.logit_bias, ddp = train_type == 'ddp', **kwargs)
        else:
            return self.loss_fn(images, texts, temperature= self.logit_scale , ddp = train_type == 'ddp', **kwargs)
class CLIP(nn.Module):
    """

    Main class for CLIP-based models.
    
    The vision model can be loaded from timm or huggingface.
    The text tower is always loaded from huggingface. There will be an adaptive projection layer 
        for the text tower (if the dimension of the text tower is different from the projection dimension).
    
    """
    def __init__(self, 
                 vision_model = 'vit_base_patch16_clip_224.dfn2b', #vit_base_patch16_clip_224.dfn2b
                 text_model = 'vinai/phobert-base-v2', 
                 vision_source = 'timm',
                 pretrain = True,
                 projection_dim = 768,
                 max_length = 64,
                 is_load = True,
                 force_text_projection = False,
                 **kwargs):
        super(CLIP, self).__init__()

        self.model_type = 'clip'
        self.loss_fn = cliploss
        self.vision_source = vision_source
        self.force_text_projection = force_text_projection
        self.train_text = False
        self.train_vision = False
        self.device = None
        self.text_model_name = text_model
        self.vision_model_name = vision_model
        self.logit_scale = nn.Parameter(torch.ones(1) * torch.log(torch.tensor(1/0.07)))
        self.max_length = max_length
        self.projection_dim = projection_dim
        if is_load:
            self.load_model(vision_model, text_model, pretrain, projection_dim)
            
        
    def load_model(self, vision_model, text_model, pretrain, projection_dim):
        print('Loading vision model')
        
        if self.vision_source == 'timm':
            self.vision_model = timm.create_model(
                vision_model,
                pretrained=pretrain,
                num_classes=0,
            )
            self.data_config = timm.data.resolve_data_config({}, model=self.vision_model)
            self.processor = timm.data.create_transform(**self.data_config, is_training=False)
        else:
            model = AutoModel.from_pretrained(vision_model)
            self.vision_model = CLIPImage(model, projection_dim)
            self.processor = CLIPImageProcessor.from_pretrained(vision_model)
            
            del model
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except:
                print('No GPU available')
        
        print(f'Number of vision model parameters: {count_parameters(self.vision_model)}')
        
        print('Loading text model')
        
        self.text_model = AutoModel.from_pretrained(text_model)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model, use_fast=True)
        print(f'Number of text model parameters: {count_parameters(self.text_model)}')
        
        if self.text_model.config.hidden_size != projection_dim or self.force_text_projection:
            self.force_text_projection = True
            self.text_projection = nn.Linear(self.text_model.config.hidden_size, projection_dim)
        
    def _setup_training(self, train_vision, train_text):
        self.train_vision = train_vision
        if not train_vision:
            print('Freezing vision model')
        for param in self.vision_model.parameters():
            param.requires_grad = train_vision
                
        # Freeze text model, only train the projection layer
        self.train_text = train_text
        if not train_text:
            print('Freezing text model')
        for param in self.text_model.parameters():
            param.requires_grad = train_text
        
        num_params = count_parameters(self)
        print(f'Number of parameters: {num_params}')
        
    
    def setup_training(self, train_vision = True, train_text = True, device = None):
        self._setup_device(device)
        self._setup_training(train_vision, train_text)
        # Freeze vision model

        
    def _setup_device(self, device = None):
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def load_checkpoint(self, checkpoint):
        self.load_state_dict(torch.load(checkpoint))
        
    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)
        
    def load_text_checkpoint(self, checkpoint):
        temp_text_model = TextEncoder(text_model = self.text_model_name,
                                 model_type = self.model_type, 
                                 projection_dim = self.projection_dim, 
                                 max_length = self.max_length, 
                                 force_text_projection = self.force_text_projection)
        temp_text_model.load_state_dict(torch.load(checkpoint))
        
        # Load weights to the current model
        with torch.no_grad():    
            self.text_model.load_state_dict(temp_text_model.text_model.state_dict())
            # self.logit_scale.load_state_dict(temp_text_model.logit_scale.state_dict())
            if self.force_text_projection:
                self.text_projection.load_state_dict(temp_text_model.text_projection.state_dict())
            # if "siglip" in self.model_type and hasattr(self, 'logit_bias'):
            #     self.logit_bias.load_state_dict(temp_text_model.logit_bias.state_dict())

        del temp_text_model
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except:
            print('No GPU available')

    def save_text_checkpoint(self, path, using_automodel = True):
        if not using_automodel:
            torch.save(self.text_model.state_dict(), os.path.join(path, 'text_model.pth'))
        else:
            self.text_model.save_pretrained(path)
        self.save_projection_checkpoint(path)
         
    def save_projection_checkpoint(self, path):
        if hasattr(self, 'text_projection'):
            torch.save(self.text_projection.state_dict(), os.path.join(path, 'text_projection.pth'))
         
    def transform_image(self, images):
        if isinstance(images, torch.Tensor):
            return images
        
        if isinstance(images, str):
            images = [images]
        
        if  all(isinstance(i, str) for i in images):
            # print("Iterating through images")
            if self.vision_source == 'timm':
                images = [open_image(i, convert_to_numpy=False) for i in images]
            else:
                images = np.array([open_image(i) for i in images])
        
        if self.vision_source == 'timm':
            return torch.stack([self.processor(image) for image in images])
        return self.processor(images = images, return_tensors='pt')

        
    def encode_image(self, images):
        images = self.transform_image(images).to(self.device)
        if self.vision_source == 'timm':
            if len(images.shape) == 3:
                images = images.unsqueeze(0)
            self.vision_model.eval()
            output = self.vision_model(images)
        
        else:
            self.vision_model.eval()
            output = self.vision_model(**images)

        emb_norm = torch.norm(output, dim=-1, keepdim=True)
        return output / (emb_norm + 1e-8)
    
    def encode_text(self, texts, result = 'mean'):
        
        
        inputs = self.tokenizer(texts, max_length=self.max_length, padding=True, truncation=True, return_tensors='pt').to(self.device)
        outputs = self.text_model(**inputs).last_hidden_state
                
        if result == 'eos':
            emb_text = outputs
            emb_text = emb_text[torch.arange(emb_text.shape[0]), torch.where(inputs['input_ids'] == self.tokenizer.eos_token_id)[1]]
        
        else:
            emb_text = mean_pooling(outputs, inputs['attention_mask'])
            
        if self.force_text_projection:
            emb_text = self.text_projection(emb_text)
        
        emb_norm = torch.norm(emb_text, dim=1, keepdim=True)
        return emb_text / (emb_norm + 1e-8)
    
    def encode(self, images, texts):
        
        image_embed = self.encode_image(images)
        text_embed = self.encode_text(texts)
        
        return image_embed, text_embed
        
    def forward(self, images, texts, train_type = 'single', **kwargs):
        image_embed = self.encode_image(images)
        text_embed = self.encode_text(texts)
        
        # Softmax loss + DDP
        return self.loss_fn(images, texts, temperature= self.logit_scale , ddp = train_type == 'ddp', **kwargs)

class SigLIP(CLIP):
    """ Similar to CLIP but with a different loss function."""
    def __init__(self, 
                 init_scale = 10,
                 init_bias = -10,
                 **kwargs):
        super(SigLIP, self).__init__(**kwargs)
        
        self.model_type = 'siglip'
        self.logit_scale = nn.Parameter(torch.ones(1) * torch.log(torch.ones(1)* init_scale))
        self.logit_bias  = nn.Parameter(torch.ones(1) * init_bias)
        self.loss_fn = sigliploss

    def forward(self, image, text, train_type = 'single', **kwargs):
        image_embed = self.encode_image(image)
        text_embed = self.encode_text(text)
        
        return self.loss_fn(image_embed, text_embed, self.logit_scale, self.logit_bias, ddp = train_type == 'ddp')
    
class LiT(CLIP):
    """ CLIP but freeze the vision model by default."""
    def __init__(self, **kwargs):
        super(LiT, self).__init__(**kwargs)
        self.model_type = 'lit'
        
    def setup_training(self, train_vision = False, train_text = True, device = None):
        self._setup_device(device)
        self._setup_training(train_vision, train_text)
        
class SigLiT(SigLIP):
    """ SigLIP but freeze the vision model by default."""
    def __init__(self, **kwargs):
        super(SigLiT, self).__init__(**kwargs)
        self.model_type = 'siglit'
        
    def setup_training(self, train_vision = False, train_text = True, device = None):
        self.setup_device(device)
        self.train_vision = train_vision
        self.train_text = train_text
    
if __name__ == "__main__":
    from PIL import Image
    from pyvi import ViTokenizer
    
    siglip = SigLIP()
    image = Image.open('data/sample/test.jpg')
    text = 'Cô gái đang sử dụng laptop'
    text = ViTokenizer.tokenize(text)
    # siglip.encode_image(image, train=False)
    text_emb = siglip.encode_text(text)
    img_emb = siglip.encode_image(image)
    
    print(text_emb @ img_emb.T)
    # print(siglip(image, text))
