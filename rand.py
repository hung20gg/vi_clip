import torch
import torch.distributed as dist
import torch.distributed
import torch.nn as nn
import os
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "54321"
    
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)


# Contrastive loss function (example implementation)
def contrastive_loss(g_image_embeds, g_text_embeds, image_embeds, text_embeds):
    # Normalize embeddings

    # Calculate logits
    img_logits = torch.matmul(image_embeds, g_text_embeds.T)
    text_logits = torch.matmul(text_embeds, g_image_embeds.T)
    
    labels = torch.arange(img_logits.size(0)) + dist.get_rank() * img_logits.size(0)
    labels = labels.to(img_logits.device)
    # Contrastive loss (cross-entropy with ground-truth diagonal matching)

    loss_img = nn.CrossEntropyLoss()(img_logits, labels)  # image-to-text loss
    loss_txt = nn.CrossEntropyLoss()(text_logits, labels)  # text-to-image loss
    
    return (loss_img + loss_txt) / 2

# Example CLIP model (simplified)

def sigliploss(image_embed, text_embed, logit_scale = 1.0, logit_bias = 0.0, ddp=False):
    
    
    labels = torch.eye(image_embed.size(0)).to(image_embed.device)
    logits = torch.matmul(image_embed, text_embed.t()) * logit_scale + logit_bias
    m1_diag1 = - torch.ones_like(logits) + 2 * labels
    
    loglik = torch.nn.functional.logsigmoid(logits * m1_diag1)
    nll = - torch.sum(loglik)
    loss = torch.mean(nll)
    
    if ddp: # DDP
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        # Go through all processes and get the image embed
        
        print(f"World size in forward model: {world_size}")
        
        for i in range(world_size):
            if i != rank: # Receive the image embed from other processes
                # Get image embed from other processes
                neighbor_image_embed = torch.empty_like(image_embed)
                torch.distributed.broadcast(neighbor_image_embed, i)
                print(neighbor_image_embed[:,:10])
                
                logits = torch.matmul(neighbor_image_embed, text_embed.t()) * logit_scale + logit_bias
                m1_diag1 = - torch.ones_like(logits) + 2 * labels
                loglik = F.logsigmoid(logits * m1_diag1)
                nll = - torch.sum(loglik)
                loss += torch.mean(nll)
                
            else: # Send the image embed to other processes
                print(f"Rank {rank} is here")
                print(image_embed[:,:10])
                torch.distributed.broadcast(image_embed, i)
    
    return loss

class CLIPModel(nn.Module):
    def __init__(self, embed_dim):
        super(CLIPModel, self).__init__()
        self.image_encoder = nn.Linear(512, embed_dim)
        self.text_encoder = nn.Linear(512, embed_dim)

    def encode_image(self, images):
        return self.image_encoder(images)

    def encode_text(self, texts):
        return self.text_encoder(texts)

# Training step with DDP and all_gather
def train(rank, world_size, model, images, texts):
    # DDP setup
    setup_ddp(rank, world_size)
    model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Move data to the correct device
    images = images.to(rank)
    texts = texts.to(rank)

    # Compute local embeddings
    image_embeds = model.module.encode_image(images)
    text_embeds = model.module.encode_text(texts)

    # Gather embeddings from all processes (all_gather)
    # gathered_image_embeds = [torch.zeros_like(image_embeds) for _ in range(world_size)]
    # gathered_text_embeds = [torch.zeros_like(text_embeds) for _ in range(world_size)]
    
    # dist.all_gather(gathered_image_embeds, image_embeds)
    # dist.all_gather(gathered_text_embeds, text_embeds)

    # # Concatenate the gathered embeddings
    # all_image_embeds = torch.cat(gathered_image_embeds, dim=0)
    # all_text_embeds = torch.cat(gathered_text_embeds, dim=0)
    
    # print("Rank", dist.get_rank())
    
    # # all_image_embeds[dist.get_rank()] = image_embeds
    # # all_text_embeds[dist.get_rank()] = text_embeds
    
    # print(all_image_embeds.shape)

    # # Compute loss using all gathered embeddings
    # loss = contrastive_loss(all_image_embeds, all_text_embeds, image_embeds, text_embeds)
    
    loss = sigliploss(image_embeds, text_embeds, ddp=True)

    # Backpropagate and update model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Rank {rank}, Loss: {loss.item()}")

# Example usage
if __name__ == "__main__":
    world_size = 2  # Number of processes (GPUs/nodes)
    embed_dim = 256  # Dimension of embeddings
    batch_size_per_node = 8

    # Example image and text input tensors (random for demonstration)
    images = torch.randn(batch_size_per_node, 512)
    texts = torch.randn(batch_size_per_node, 512)

    # Initialize a simple CLIP-like model
    model = CLIPModel(embed_dim)

    # Simulate training on each node with multiple processes
    torch.multiprocessing.spawn(
        train,
        args=(world_size, model, images, texts),
        nprocs=world_size,
    )
