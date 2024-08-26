import torch 
import torch.nn as nn
import torch.nn.functional as F
from .utils import all_gather_default

""" 
All gather version of sigliploss

"""
def sigliploss_allgather(image_embed, text_embed, logit_scale = 1.0, logit_bias = 0.0, **kwargs):
    
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    
    tensor_list = [torch.empty_like(image_embed) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, image_embed)
    loss = None
    
    for i in range(world_size):
        neighbor_image_embed = tensor_list[i]
        logits = torch.matmul(neighbor_image_embed, text_embed.t()) * logit_scale + logit_bias
        labels = torch.eye(neighbor_image_embed.size(0)).to(text_embed.device)
        m1_diag1 = - torch.ones_like(logits) + 2 * labels * (1 if i == rank else 0)
        loglik = torch.nn.functional.logsigmoid(logits * m1_diag1)
        
        nll = - torch.sum(loglik)
        if loss is None:
            loss = torch.mean(nll)
        else:
            loss += torch.mean(nll)
    
    torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
    return loss


def sigliploss(image_embed, text_embed, logit_scale = 1.0, logit_bias = 0.0, ddp=False, **kwargs):
    
    labels = torch.eye(image_embed.size(0)).to(image_embed.device)
    logits = torch.matmul(image_embed, text_embed.t()) * logit_scale + logit_bias
    m1_diag1 = - torch.ones_like(logits) + 2 * labels
    
    loglik = F.logsigmoid(logits * m1_diag1)
    nll = - torch.sum(loglik)
    loss = torch.mean(nll)
    
    if ddp: # DDP
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        
        # I dont know is this code optimized
        # Send local embeddings to all processes
        torch.distributed.broadcast(image_embed, rank)
        # Go through all processes and get the image embed
        for i in range(world_size):
            if i != rank:
                
                # Get image embed from other processes
                neighbor_image_embed = torch.empty_like(image_embed)
                torch.distributed.broadcast(neighbor_image_embed, i)
                
                logits = torch.matmul(neighbor_image_embed, text_embed.t()) * logit_scale + logit_bias

                # No need to calculate the diagonal 1
                loglik = F.logsigmoid( - logits)
                nll = - torch.sum(loglik)
                loss += torch.mean(nll)
        torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
    return loss

def cliploss(image_embed, text_embed, temperature = 1/0.07, ddp = False, **kwargs):
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    # Do contrastive loss on local - global embeddings (diag 1 shifted by the current rank)
    if ddp:
        rank = torch.distributed.get_rank() 
        
        all_image_embed = all_gather_default(image_embed)
        all_text_embed = all_gather_default(text_embed)
        
        logits_image = torch.matmul(image_embed, all_text_embed.t()) * torch.exp(temperature)
        logits_text = torch.matmul(text_embed, all_image_embed.t()) * torch.exp(temperature)
        label = torch.arange(all_image_embed.size(0)) + rank * all_image_embed.size(0)
        label = label.to(all_image_embed.device)
    
    else: # Do contrastive loss on local embeddings
        logits_image = torch.matmul(image_embed, text_embed.t()) * torch.exp(temperature)
        logits_text = torch.matmul(text_embed, image_embed.t()) * torch.exp(temperature)
        label = torch.arange(image_embed.size(0)).to(image_embed.device)
        
    loss = (loss_img(logits_image, label) + loss_txt(logits_text, label)) / 2
    
    return loss

if __name__ == '__main__':
    image_embed = torch.randn(3, 4)
    text_embed = torch.randn(3, 4)
    loss = sigliploss(image_embed, text_embed)
    print(loss)