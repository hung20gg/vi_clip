import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import torch.distributed as dist

# https://github.com/zsnoob/EfficientDDP-4-Contrastive-Train/blob/main/ED4CT/AllGather.py
# With optimizing your similarity matrix result from [global, global] to [local, global]
class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):
        
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = rank
        ctx.local_batch_size = tensor.shape[0]
        return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        # Reduce gradients prior to gradient bucket to avoid behavior mismatches with non-distributed training
        dist.all_reduce(grad_output, op=dist.ReduceOp.AVG)

        return (
            grad_output[ctx.local_batch_size * ctx.rank: ctx.local_batch_size * (ctx.rank + 1)],
            None,
        )
        
# https://github.com/IDEA-Research/DisCo-CLIP/blob/main/disco/gather.py   
class DisCoGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor):
        if not torch.distributed.is_initialized():
            raise "torch.distributed is not initialized"

        world_size = torch.distributed.get_world_size()
        ctx.bs = tensor.shape[0]
        ctx.rank = torch.distributed.get_rank()

        gathered_tensors = [
            torch.zeros_like(tensor) for _ in range(world_size)
        ]
        torch.distributed.all_gather(gathered_tensors, tensor)

        gathered_tensors = torch.cat(gathered_tensors, dim=0)
        gathered_tensors.requires_grad_(True)

        return gathered_tensors

    @staticmethod
    def backward(ctx, grad_output):
        torch.distributed.all_reduce(grad_output, op=dist.ReduceOp.AVG)
        return grad_output[ctx.bs*ctx.rank:ctx.bs*(ctx.rank+1)]

# the gather method used in https://amsword.medium.com/gradient-backpropagation-with-torch-distributed-all-gather-9f3941a381f8
def all_gather_default(tensor, train = True):
    world_size = torch.distributed.get_world_size()
    print(f"World size in forward model: {world_size}")
    # with torch.no_grad():
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, tensor)
    
    # All gather is not needed for evaluation
    if train:
        tensor_list[torch.distributed.get_rank()] = tensor
        
    tensor_list = torch.cat(tensor_list, dim=0)
    return tensor_list

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


def print_detail(tensor: torch.Tensor, name: str):
    print(f"Name: {name}")
    print(f"Shape: {tensor.shape}")
    print(f"Device: {tensor.device}")
    print(f"Requires_grad: {tensor.requires_grad}")
    print(f"Data type: {tensor.dtype}")
    

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
    # from PIL import Image
    # from transformers import AutoTokenizer, AutoModel, AutoProcessor
    
    # model = AutoModel.from_pretrained('openai/clip-vit-base-patch16').to('cuda')
    # processor = AutoProcessor.from_pretrained('openai/clip-vit-base-patch16')
    # text_model = CLIPText(model)
    # vision_model = CLIPImage(model)
    # del model
    # torch.cuda.empty_cache()
    
    # image = Image.open('..\\sample\\Donald-Trump.jpg')
    # text = 'A photo of Donald Trump'
    # image = processor(images = image, return_tensors = 'pt', padding = True, truncation = True).to('cuda')
    # output = vision_model(**image)
    # print(output/ torch.norm(output, dim = -1, keepdim = True))
    x = torch.randn(2, 768)
    AllGather.apply(x)
    