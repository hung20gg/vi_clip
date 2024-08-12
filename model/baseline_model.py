import torch 
import numpy as np
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from .utils import open_image, CLIPImage, CLIPText
from .lossfn import cliploss, sigliploss
from .model import CLIP
import gc

class BaselineCLIP(CLIP):
    """
        Implementation of the Baseline CLIP model.

    """
    def __init__(self, clip_model, **kwargs):
        super().__init__(is_load= False, **kwargs)
        self.model_name = clip_model
        self.vision_source = 'huggingface'
        
        # Maximum length of the clip text. 
        self.max_length = 77
        
        model = AutoModel.from_pretrained(clip_model)
        self.text_model = CLIPText(model)
        self.vision_model = CLIPImage(model)
        del model
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except:
            print('No GPU available')
            
        self.processor = AutoProcessor.from_pretrained(clip_model)
        self.tokenizer = AutoTokenizer.from_pretrained(clip_model)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
class BaselineSigLIP(BaselineCLIP):
    """
        Implementation of the Baseline SigLIP model.
    """
    def __init__(self, clip_model, **kwargs):
        super().__init__(is_load= False, clip_model=clip_model, **kwargs)
        self.loss_fn = sigliploss