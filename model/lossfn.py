import torch 
import torch.nn as nn
import torch.nn.functional as F

def sigliploss(image_embed, text_embed, logit_scale = 1.0, logit_bias = 0.0):
    
    label = torch.eye(image_embed.size(0)).to(image_embed.device)
    logits = torch.matmul(image_embed, text_embed.t()) * logit_scale + logit_bias
    
    m1_diag1 = - torch.ones_like(logits) + 2 * label
    
    loglik = F.logsigmoid(logits * m1_diag1)
    nll = - torch.sum(loglik)
    loss = torch.mean(nll)
    
    return loss

def cliploss(image_embed, text_embed):
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

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