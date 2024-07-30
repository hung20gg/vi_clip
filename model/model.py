import torch
import torch.nn as nn
from .lossfn import sigliploss, cliploss
from transformers import AutoTokenizer, AutoModel, AutoProcessor
import timm
from .utils import mean_pooling, count_parameters


class CLIP(nn.Module):
    def __init__(self, 
                 vision_model = 'vit_base_patch16_clip_224.dfn2b', #vit_base_patch16_clip_224.dfn2b
                 text_model = 'vinai/phobert-base-v2', 
                 vision_source = 'timm',
                 pretrain = True,
                 device = None):
        super(CLIP, self).__init__()

        
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.loss_fn = cliploss
        
        print(f"Using device: {self.device}")
        
        print('Loading vision model')
        if vision_source == 'timm':
            self.vision_model = timm.create_model(
                vision_model,
                pretrained=pretrain,
                num_classes=0,
            )
            self.data_config = timm.data.resolve_data_config({}, model=self.vision_model)
            self.processor = timm.data.create_transform(**self.data_config, is_training=False)
        else:
            self.vision_model = AutoModel.from_pretrained(vision_model)
            self.processor = AutoProcessor.from_pretrained(vision_model)
        
        print(f'Number of vision model parameters: {count_parameters(self.vision_model)}')
        
        self.train_text = False
        self.train_vision = False
        
        print('Loading text model')
        self.text_model = AutoModel.from_pretrained(text_model)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model, use_fast=True)
        print(f'Number of text model parameters: {count_parameters(self.text_model)}')
    
    def setup_training(self, train_vision = True, train_text = True):
        self.train_vision = train_vision
        self.train_text = train_text
    
    def load_checkpoint(self, checkpoint):
        self.load_state_dict(checkpoint['model_state_dict'])
        
    def transform_image(self, image):
        if isinstance(image, torch.Tensor):
            return image
        return self.processor(image)
        
    def encode_image(self, image, train = False):
        image = self.transform_image(image).to(self.device)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        if self.train_vision or train:
            self.vision_model.train()
            output = self.vision_model(image)
        else:
            self.vision_model.eval()
            with torch.no_grad():
                output = self.vision_model(image)

        emb_norm = torch.norm(output, dim=1, keepdim=True)
        return output / (emb_norm + 1e-8)
    
    def encode_text(self, text, result = 'mean', train = False):
        
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
        
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
                 vision_model = 'vit_base_patch16_siglip_224', #vit_base_patch16_clip_224.dfn2b
                 text_model = 'vinai/phobert-base-v2', 
                 vision_source = 'timm',
                 pretrain = True,
                 device = None,
                 init_scale = 10,
                 init_bias = -10):
        super(SigLIP, self).__init__(vision_model, text_model, vision_source, pretrain, device)
        
        self.logit_scale = nn.Parameter(torch.ones(1) * torch.log(torch.ones(1)* init_scale))
        self.logit_bias  = nn.Parameter(torch.ones(1) * init_bias)
        self.loss_fn = sigliploss

    def forward(self, image, text):
        image_embed = self.encode_image(image)
        text_embed = self.encode_text(text)
        
        return self.loss_fn(image_embed, text_embed, self.logit_scale, self.logit_bias)
    
class LiT(CLIP):
    def __init__(self, 
                 vision_model = 'vit_base_patch16_clip_224.openai', #vit_base_patch16_clip_224.dfn2b
                 text_model = 'vinai/phobert-base-v2', 
                 vision_source = 'timm',
                 pretrain = True,
                 device = None):
        super(LiT, self).__init__(vision_model, text_model, vision_source, pretrain, device)

    def setup_training(self, train_vision = False, train_text = True):
        self.train_vision = train_vision
        self.train_text = train_text
        
class SigLiT(SigLIP):
    def __init__(self, 
                 vision_model = 'vit_base_patch16_siglip_224', #vit_base_patch16_clip_224.dfn2b
                 text_model = 'vinai/phobert-base-v2', 
                 vision_source = 'timm',
                 pretrain = True,
                 device = None,
                 init_scale = 10,
                 init_bias = -10):
        super(SigLiT, self).__init__(vision_model, text_model, vision_source, pretrain, device, init_scale, init_bias)
        
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
