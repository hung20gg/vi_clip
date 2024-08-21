import torch 
import numpy as np
from transformers import AutoModel, CLIPImageProcessor, AutoTokenizer
from .utils import open_image, CLIPImage, CLIPText
from .lossfn import cliploss, sigliploss
from .model import CLIP
import gc

class BaselineCLIP(CLIP):
    """
        Implementation of the Baseline CLIP model.

    """
    def __init__(self, clip_model, max_length = 77, **kwargs):
        super().__init__(is_load= False, **kwargs)
        self.model_name = clip_model
        self.vision_source = 'huggingface'
        
        # Maximum length of the clip text. 
        self.max_length = max_length
        
        model = AutoModel.from_pretrained(clip_model)
        self.text_model = CLIPText(model)
        self.vision_model = CLIPImage(model)
        del model
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except:
            print('No GPU available')
            
        self.processor = CLIPImageProcessor.from_pretrained(clip_model)
        self.tokenizer = AutoTokenizer.from_pretrained(clip_model)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def encode_text(self, text, result = 'eos'):
        inputs = self.tokenizer(text, max_length=self.max_length, padding=True, truncation=True, return_tensors='pt').to(self.device)
    
        emb_text = self.text_model(**inputs)

        emb_norm = torch.norm(emb_text, dim=1, keepdim=True)
        return emb_text / (emb_norm + 1e-8)
        
class BaselineSigLIP(BaselineCLIP):
    """
        Implementation of the Baseline SigLIP model.
    """
    def __init__(self, clip_model, **kwargs):
        super().__init__(is_load= False, clip_model=clip_model, **kwargs)
        self.loss_fn = sigliploss
        self.max_length = min(self.max_length, 64)
        
        
# CLIP model change the embedding layer to new vocab.