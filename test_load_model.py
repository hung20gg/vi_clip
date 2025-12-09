from transformers import AutoModel

import torch

model = AutoModel.from_pretrained("vinai/phobert-base-v2")
model.eval()