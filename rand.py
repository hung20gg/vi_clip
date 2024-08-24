import torch
import torch.distributed as dist
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
def contrastive_loss(image_embeds, text_embeds):
    # Normalize embeddings
    image_embeds = nn.functional.normalize(image_embeds, p=2, dim=1)
    text_embeds = nn.functional.normalize(text_embeds, p=2, dim=1)
    
    # Calculate logits
    logits = image_embeds @ text_embeds.T
    
    # Contrastive loss (cross-entropy with ground-truth diagonal matching)
    labels = torch.arange(logits.size(0)).to(logits.device)
    loss_img = nn.CrossEntropyLoss()(logits, labels)  # image-to-text loss
    loss_txt = nn.CrossEntropyLoss()(logits.T, labels)  # text-to-image loss
    
    return (loss_img + loss_txt) / 2

# Example CLIP model (simplified)
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

    # Compute loss using all gathered embeddings
    loss = contrastive_loss(all_image_embeds, all_text_embeds)

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
