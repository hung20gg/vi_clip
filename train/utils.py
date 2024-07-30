from ..model.model import CLIP, SigLIP, LiT, SigLiT
from torch.utils.data import DataLoader, Dataset
from .dataloader import ImageCaptionDataset, CLIPSampler

def build_model(model_args):
    
    text_encoder = model_args['text_encoder']
    image_encoder = model_args['image_encoder']
    
    model_type = model_args['model_type']
    
    model = None
    
    if model_type.lower() == 'clip':
        model = CLIP(text_encoder = text_encoder, image_encoder = image_encoder)
    elif model_type.lower() == 'siglip':
        model = SigLIP(text_encoder = text_encoder, image_encoder = image_encoder)
    elif model_type.lower() == 'lit':
        model = LiT(text_encoder = text_encoder, image_encoder = image_encoder)
    elif model_type.lower() == 'siglit':
        model = SigLiT(text_encoder = text_encoder, image_encoder = image_encoder)
    return model

def get_dataloader(train_args, train = True):
    data = train_args['dataset']
    batch_size = train_args['batch_size']
    num_workers = train_args['num_workers']
    
    dataset = ImageCaptionDataset(data)
    if train:
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    else:
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers)
    
    return dataloader