from ..model import CLIP, SigLIP, LiT, SigLiT
from torch.utils.data import DataLoader
from .dataloader import ImageCaptionDataset, CLIPSampler, CrossLingualDataset, mCLIPDataset

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

def get_dataloader(train_args, model_args, train = True):
    datasets = train_args['dataset']
    training_objective = model_args['model_type']
    batch_size = train_args['batch_size']
    num_workers = train_args['num_workers']
    sampler = None
    dataloaders = []
    
    if isinstance(datasets, str):
        datasets = [datasets]
    
    for data in datasets:
        if training_objective in ['clip','siglip','lit','siglit']:
            dataset = ImageCaptionDataset(data)
            sampler = CLIPSampler(duplicate_id = 0, batch_size = batch_size)
        elif training_objective == 'crosslingual':
            dataset = CrossLingualDataset(data)
        else:
            dataset = mCLIPDataset(data)
        
        if train:
            dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, sampler=sampler, num_workers = num_workers)
        else:
            dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers)
        
        dataloaders.append(dataloader)

    return dataloaders