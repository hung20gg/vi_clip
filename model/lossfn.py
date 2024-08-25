import torch 
import torch.nn as nn
import torch.nn.functional as F

def sigliploss(image_embed, text_embed, logit_scale = 1.0, logit_bias = 0.0, ddp=False):
    
    
    labels = torch.eye(image_embed.size(0)).to(image_embed.device)
    logits = torch.matmul(image_embed, text_embed.t()) * logit_scale + logit_bias
    m1_diag1 = - torch.ones_like(logits) + 2 * labels
    
    loglik = F.logsigmoid(logits * m1_diag1)
    nll = - torch.sum(loglik)
    loss = torch.mean(nll)
    
    if ddp: # DDP
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        # Go through all processes and get the image embed
        for i in range(world_size):
            if i != rank:
                # Get image embed from other processes
                neighbor_image_embed = torch.empty_like(image_embed)
                torch.distributed.broadcast(neighbor_image_embed, i)
                
                logits = torch.matmul(neighbor_image_embed, text_embed.t()) * logit_scale + logit_bias
                m1_diag1 = - torch.ones_like(logits) + 2 * labels
                loglik = F.logsigmoid(logits * m1_diag1)
                nll = - torch.sum(loglik)
                loss += torch.mean(nll)
    
    return loss

def cliploss(image_embed, text_embed, all_image_embed = None, all_text_embed = None):
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    # Do contrastive loss on local - global embeddings (diag 1 shifted by the current rank)
    if all_image_embed is not None and all_text_embed is not None:
        logits_image = torch.matmul(image_embed, all_text_embed.t())
        logits_text = torch.matmul(text_embed, all_image_embed.t())
        label = torch.arange(all_image_embed.size(0)) + torch.distributed.get_rank() * all_image_embed.size(0)
        label = label.to(all_image_embed.device)
    
    else: # Do contrastive loss on local embeddings
        logits_image = torch.matmul(image_embed, text_embed.t())
        logits_text = torch.matmul(text_embed, image_embed.t())
        label = torch.arange(image_embed.size(0)).to(image_embed.device)
        
    loss = (loss_img(logits_image, label) + loss_txt(logits_text, label)) / 2
    
    return loss

if __name__ == '__main__':
    image_embed = torch.randn(3, 4)
    text_embed = torch.randn(3, 4)
    loss = sigliploss(image_embed, text_embed)
    print(loss)