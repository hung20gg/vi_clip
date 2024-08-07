import torch
import torch.nn as nn
import numpy as np
from .lossfn import sigliploss, cliploss
from transformers import AutoTokenizer, AutoModel, AutoProcessor, CLIPImageProcessor
import timm
from .utils import mean_pooling, count_parameters, open_image, CLIPImage, CLIPText
import gc


class CLIP(nn.Module):
    def __init__(self, 
                 vision_model = 'vit_base_patch16_clip_224.dfn2b', #vit_base_patch16_clip_224.dfn2b
                 text_model = 'vinai/phobert-base-v2', 
                 vision_source = 'timm',
                 pretrain = True,
                 projection_dim = 768,
                 max_length = 64,
                 **kwargs):
        super(CLIP, self).__init__()

        self.loss_fn = cliploss

        print('Loading vision model')
        self.vision_source = vision_source
        if vision_source == 'timm':
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
        
        print(f'Number of vision model parameters: {count_parameters(self.vision_model)}')
        
        self.train_text = False
        self.train_vision = False
        
        print('Loading text model')
        self.text_model = AutoModel.from_pretrained(text_model)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model, use_fast=True)
        print(f'Number of text model parameters: {count_parameters(self.text_model)}')
        
        self.max_length = max_length
        self.to(self.device)
    
    def setup_training(self, train_vision = True, train_text = True):
        self.train_vision = train_vision
        if not train_vision:
            self.vision_model.requires_grad = False
        self.train_text = train_text
    
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
         
    def transform_image(self, image):
        if isinstance(image, torch.Tensor):
            return image.to(self.device)
        
        if isinstance(image, list) and all(isinstance(i, str) for i in image):
            image = np.array([open_image(i) for i in image])
        
        if self.vision_source == 'timm':
            return self.processor(image).to(self.device)
        return self.processor(image, return_tensors='pt').to(self.device)

        
    def encode_image(self, image, train = False):
        image = self.transform_image(image).to(self.vision_model.device)
        if self.vision_source == 'timm':
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            if self.train_vision or train:
                self.vision_model.train()
                output = self.vision_model(image)
            else:
                self.vision_model.eval()
                with torch.no_grad():
                    output = self.vision_model(image)
        
        else:
            if self.train_vision or train:
                self.vision_model.train()
                output = self.vision_model(**image)
            else:
                self.vision_model.eval()
                with torch.no_grad():
                    output = self.vision_model(**image)

        emb_norm = torch.norm(output, dim=1, keepdim=True)
        return output / (emb_norm + 1e-8)
    
    def encode_text(self, text, result = 'mean', train = False):
        
        inputs = self.tokenizer(text, max_length=self.max_length, padding=True, truncation=True, return_tensors='pt').to(self.text_model.device)
        
        if self.train_text or train:
            self.text_model.train()
            outputs = self.text_model(**inputs).last_hidden_state
        else:
            self.text_model.eval()
            with torch.no_grad():
                outputs = self.text_model(**inputs).last_hidden_state
                
        if result == 'eos':
            emb_text = outputs
            emb_text = emb_text[torch.arange(emb_text.shape[0]), torch.where(inputs['input_ids'] == self.tokenizer.eos_token_id)[1]]
        
        else:
            emb_text = mean_pooling(outputs, inputs['attention_mask'])
        
        emb_norm = torch.norm(emb_text, dim=1, keepdim=True)
        return emb_text / (emb_norm + 1e-8)
        
    def forward(self, image, text):
        image_embed = self.encode_image(image)
        text_embed = self.encode_text(text)
        
        return self.loss_fn(image_embed, text_embed)
    

class SigLIP(CLIP):
    def __init__(self, 
                 init_scale = 10,
                 init_bias = -10,
                 **kwargs):
        super(SigLIP, self).__init__(**kwargs)
        
        self.logit_scale = nn.Parameter(torch.ones(1) * torch.log(torch.ones(1)* init_scale))
        self.logit_bias  = nn.Parameter(torch.ones(1) * init_bias)
        self.loss_fn = sigliploss

    def forward(self, image, text):
        image_embed = self.encode_image(image)
        text_embed = self.encode_text(text)
        
        return self.loss_fn(image_embed, text_embed, self.logit_scale, self.logit_bias)
    
class LiT(CLIP):
    def __init__(self, **kwargs):
        super(LiT, self).__init__(**kwargs)
        
    def setup_training(self, train_vision = False, train_text = True):
        self.train_vision = train_vision
        self.train_text = train_text
        
class SigLiT(SigLIP):
    def __init__(self, **kwargs):
        super(SigLiT, self).__init__(**kwargs)
        
    def setup_training(self, train_vision = False, train_text = True):
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
