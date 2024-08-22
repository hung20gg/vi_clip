import torch
import torch.nn as nn
from PIL import Image
import numpy as np


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def open_image(image, size = 224, convert_to_numpy = True):
    image = Image.open(image).convert('RGB')
    if image.width != size or image.height != size:
        image = image.resize((size, size))
    
    # Convert to numpy
    if convert_to_numpy:
        image = np.array(image)
        if image.shape == 2:
            image = np.stack([image, image, image], axis = -1)
        if image.shape[2] == 4:
            image = image[:,:,:3]
    return image

class CLIPText(nn.Module):
    """ 
        Get the text tower model from CLIP/SigLIP
    """
    def __init__(self, model):
        super(CLIPText, self).__init__()
        self.text_model = model.text_model

        if hasattr(model, 'text_projection'):
            self.text_projection = model.text_projection
        else:
            self.text_projection = nn.Identity()

    def forward(self, **x):
        text_outputs = self.text_model(**x)[1]
        return self.text_projection(text_outputs)
    
class CLIPImage(nn.Module):
    """ 
        Get the image tower model from CLIP/SigLIP
    """
    def __init__(self, model):
        super(CLIPImage, self).__init__()
        self.vision_model = model.vision_model
        
        if hasattr(model, 'visual_projection'):
            self.visual_projection = model.visual_projection
        else:
            self.visual_projection = nn.Identity()
        
    def forward(self, **x):
        vision_outputs = self.vision_model(**x)[1]
        # vision_outputs = vision_outputs[1]
        return self.visual_projection(vision_outputs)
    
if __name__ == '__main__':
    from PIL import Image
    from transformers import AutoTokenizer, AutoModel, AutoProcessor
    
    model = AutoModel.from_pretrained('openai/clip-vit-base-patch16').to('cuda')
    processor = AutoProcessor.from_pretrained('openai/clip-vit-base-patch16')
    text_model = CLIPText(model)
    vision_model = CLIPImage(model)
    del model
    torch.cuda.empty_cache()
    
    image = Image.open('..\\sample\\Donald-Trump.jpg')
    text = 'A photo of Donald Trump'
    image = processor(images = image, return_tensors = 'pt', padding = True, truncation = True).to('cuda')
    output = vision_model(**image)
    print(output/ torch.norm(output, dim = -1, keepdim = True))
    
    