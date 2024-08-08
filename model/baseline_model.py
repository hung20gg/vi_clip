import torch 
import numpy as np
from transformers import AutoModel, AutoProcessor
from .utils import open_image

class BaselineCLIP:
    def __init__(self, model_name, device = None):
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
    
       
        
    def encode(self, images, texts):
        if all(isinstance(i, str) for i in images):
            image = np.array([open_image(i) for i in image])
        inputs = self.processor(text = texts, images = images , return_tensors = 'pt', padding = True, truncation = True).to(self.device)
        outputs = self.model(**inputs)
        
        text_emb = outputs.text_embeds.pooler_output
        image_emb = outputs.image_embeds.pooler_output
        
        return image_emb, text_emb