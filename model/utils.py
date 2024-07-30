import torch
import torch.nn as nn

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class CLIPText(nn.Module):
    def __init__(self, model):
        super(CLIPText, self).__init__()
        layers = [model.text_model]
        if hasattr(model, 'text_projection'):
            layers.append(model.text_projection)

        self.text = nn.ModuleList(layers)
        
    def forward(self, x):
        return self.text(x)
    
class CLIPImage(nn.Module):
    def __init__(self, model, projection_dim = 768):
        super(CLIPImage, self).__init__()
        layers = [model.vision_model]
        if hasattr(model, 'vision_projection') and model.projection_dim == projection_dim:
            layers.append(model.vision_projection)
        self.image = nn.ModuleList(layers)
        
    def forward(self, x):
        return self.image(x)

if __name__ == '__main__':
    x = torch.randn(3, 7, 768)
    y = mean_pooling(x, torch.ones(3, 7))
    print(y.shape)