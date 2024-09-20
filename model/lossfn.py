import torch 
import torch.nn as nn
import torch.nn.functional as F
from .utils import all_gather_default, DisCoGather

""" 
All gather version of sigliploss

"""
def sigliploss_allgather(image_embed, text_embed, logit_scale = 1.0, logit_bias = 0.0, all_gather_implement = 'disco', require_grad = False, **kwargs):
    
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    
    # Only need to gather the gradients of one embedding, in this case is the image embeddings
    if not require_grad: # No need to gather the gradients of the image embeddings
        tensor_list = [torch.empty_like(image_embed) for _ in range(world_size)]
        torch.distributed.all_gather(tensor_list, image_embed)
    else:
        if all_gather_implement == 'disco':
            tensor_list = DisCoGather.apply(image_embed)
        else:
            tensor_list = all_gather_default(image_embed, train=True)
    
    loss = None
    
    for i in range(world_size):
        if not require_grad:
            neighbor_image_embed = tensor_list[i]
        else:
            bs = image_embed.size(0)
            neighbor_image_embed = tensor_list[i*bs:(i+1)*bs, :]
            
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


def sigliploss_broadcast(image_embed, text_embed, logit_scale = 1.0, logit_bias = 0.0, ddp=False, **kwargs):
    
    labels = torch.eye(image_embed.size(0)).to(image_embed.device)
    logits = torch.matmul(image_embed, text_embed.t()) * logit_scale + logit_bias
    m1_diag1 = - torch.ones_like(logits) + 2 * labels
    
    loglik = F.logsigmoid(logits * m1_diag1)
    nll = - torch.sum(loglik)
    loss = torch.mean(nll)
    
    if ddp: # DDP
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        
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

def sigliploss(image_embed, text_embed, logit_scale = 1.0, logit_bias = 0.0, ddp = False, all_gather = True, all_gather_implement = 'disco', require_gard = False):
    
    all_gather = all_gather and ddp
    
    if all_gather:
        return sigliploss_allgather(image_embed, text_embed, logit_scale = logit_scale, logit_bias = logit_bias, all_gather_implement = all_gather_implement, require_grad = require_gard)
    else:
        return sigliploss_broadcast(image_embed, text_embed, logit_scale = logit_scale, logit_bias = logit_bias, ddp = ddp)
    

def cliploss(image_embed, text_embed, temperature = 1/0.07, ddp = False, require_grad_image = False, require_grad_text = True, all_gather_implement = 'disco', **kwargs):
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    # Do contrastive loss on local - global embeddings (diag 1 shifted by the current rank)
    if ddp:
        # Picking the implementation of all gather
        rank = torch.distributed.get_rank() 
        if all_gather_implement == 'disco':
            all_image_embed = DisCoGather.apply(image_embed)
            all_text_embed = DisCoGather.apply(text_embed)
        else:   
            all_image_embed = all_gather_default(image_embed, train=require_grad_image)
            all_text_embed = all_gather_default(text_embed, train=require_grad_text)
        
        # Calculate the logits
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