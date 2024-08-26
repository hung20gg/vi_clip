import torch
import torch.distributed as dist
import torch.distributed
import torch.nn as nn
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import time


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

def sigliploss(image_embed, text_embed, logit_scale = 1.0, logit_bias = 0.0, ddp=False, all_gather = True):
    print(f"DDP: {ddp}, All gather: {all_gather}")
    if ddp == False and all_gather == True:
        raise ValueError("All gather is only available in DDP mode")
    
    if ddp and all_gather: # DDP + All gather 
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        
        print(f"World size in forward model: {world_size}")
        tensor_list = [torch.empty_like(image_embed) for _ in range(world_size)]
        torch.distributed.all_gather(tensor_list, image_embed)
        loss = None
        
        for i in range(world_size):

            neighbor_image_embed = tensor_list[i]
            # print(f"______________\nRank {rank} received image embed from rank {i}, shape: {neighbor_image_embed.shape}",neighbor_image_embed[:,:10])
            
            logits = torch.matmul(neighbor_image_embed, text_embed.t()) * logit_scale + logit_bias
            labels = torch.eye(neighbor_image_embed.size(0)).to(text_embed.device)
            m1_diag1 = - torch.ones_like(logits) + 2 * labels * (1 if i == rank else 0)
            loglik = torch.nn.functional.logsigmoid(logits * m1_diag1)
            
            nll = - torch.sum(loglik)
            print(all_gather, f"Rank {rank} received image embed from rank {i}, loss: ", torch.mean(nll))
            
            if loss is None:
                loss = torch.mean(nll)
            else:
                loss += torch.mean(nll)
        
        return loss 


    labels = torch.eye(image_embed.size(0)).to(image_embed.device)
    logits = torch.matmul(image_embed, text_embed.t()) * logit_scale + logit_bias
    m1_diag1 = - torch.ones_like(logits) + 2 * labels
    
    loglik = torch.nn.functional.logsigmoid(logits * m1_diag1)
    nll = - torch.sum(loglik)
    loss = torch.mean(nll)
    
    
    if ddp and not all_gather: # DDP go through all processes and get the image embed
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        # Go through all processes and get the image embed
        
        print(f"World size in forward model: {world_size}")
        print(all_gather, f"Rank {rank} received image embed from rank {rank}, loss: ", loss)
        torch.distributed.broadcast(image_embed, rank)
        
        for i in range(world_size):
            if i != rank: # Receive the image embed from other processes
                # Get image embed from other processes
                neighbor_image_embed = torch.empty_like(image_embed)
                torch.distributed.broadcast(neighbor_image_embed, i)
                # print(f"______________\nRank {rank} received image embed from rank {i}, shape: {neighbor_image_embed.shape}",neighbor_image_embed[:,:10])
                
                logits = torch.matmul(neighbor_image_embed, text_embed.t()) * logit_scale + logit_bias
                
                # No need to calculate the diagonal 1
                loglik = torch.nn.functional.logsigmoid(- logits)
                nll = - torch.sum(loglik)
                loss += torch.mean(nll)
                
                print(all_gather, f"Rank {rank} received image embed from rank {i}, loss: ", torch.mean(nll))
                # print('2 neighbors', (image_embed + neighbor_image_embed)[:,:10])
                
            # else: # Send the image embed to other processes
            #     print(f"___________\nRank {rank} is here, shape: {image_embed.shape}", image_embed[:,:10])
    
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
def train(rank, world_size, model, images, texts, all_gather=False, loss_ = 'siglip'):
    # DDP setup
    setup_ddp(rank, world_size)
    model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Move data to the correct device
    images = images.to(rank)
    texts = texts.to(rank)

    # Compute local embeddings
    print("Rank",rank, "Embedding image and text")
    image_embeds = model.module.encode_image(images)
    text_embeds = model.module.encode_text(texts)
    
    if rank == 1:
        image_embeds = image_embeds * 2
        text_embeds = text_embeds * 2
        
    # print(f"_________\nRank",rank, image_embeds[:,:10])

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
    model.eval()
    with torch.no_grad():
        if loss_ == 'siglip':
            loss = sigliploss(image_embeds, text_embeds, ddp=True, all_gather=all_gather)
            time.sleep(10)
            
            loss2 = sigliploss(image_embeds, text_embeds, ddp=True, all_gather=not all_gather)
            print("Loss 2", loss2)
            print("Loss 1", loss)
        else:
            gathered_image_embeds = [torch.zeros_like(image_embeds) for _ in range(world_size)]
            gathered_text_embeds = [torch.zeros_like(text_embeds) for _ in range(world_size)]
            
            dist.all_gather(gathered_image_embeds, image_embeds)
            dist.all_gather(gathered_text_embeds, text_embeds)

            # Concatenate the gathered embeddings
            all_image_embeds = torch.cat(gathered_image_embeds, dim=0)
            all_text_embeds = torch.cat(gathered_text_embeds, dim=0)
            
            print("Rank", dist.get_rank())
            
            # all_image_embeds[dist.get_rank()] = image_embeds
            # all_text_embeds[dist.get_rank()] = text_embeds
            
            print(all_image_embeds.shape)
            loss = contrastive_loss(all_image_embeds, all_text_embeds, image_embeds, text_embeds)
        
    # loss = loss.sum()
    # print("Total loss", loss)

    # Backpropagate and update model
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()

    print(f"Rank {rank}, Loss: {loss.sum().item()}")

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Script')
    world_size = 2  # Number of processes (GPUs/nodes)
    embed_dim = 256  # Dimension of embeddings
    batch_size_per_node = 8
    all_gather = False
    type_ = 'siglip'
    
    parser.add_argument('--all_gather', type=bool, default=all_gather, help='All gather embeddings')
    parser.add_argument('--type', type=str, default=type_, help='Type of loss function')
    args = parser.parse_args()

    # Example image and text input tensors (random for demonstration)
    images = torch.randn(batch_size_per_node, 512)
    texts = torch.randn(batch_size_per_node, 512)

    # Initialize a simple CLIP-like model
    model = CLIPModel(embed_dim)

    # Simulate training on each node with multiple processes
    torch.multiprocessing.spawn(
        train,
        args=(world_size, model, images, texts, args.all_gather, args.type),
        nprocs=world_size,
    )
