from ..model import CLIP, SigLIP, LiT, SigLiT, CrossLingual, mCLIP
import os
import pandas as pd
from torch.utils.data import DataLoader
from .dataloader import ImageCaptionDataset, CLIPSampler, CrossLingualDataset, mCLIPDataset

def build_model(model_args):
        
    model_type = model_args['model_type']
    
    model = None
    
    if model_type.lower() == 'clip':
        model = CLIP(**model_args)
    elif model_type.lower() == 'siglip':
        model = SigLIP(**model_args)
    elif model_type.lower() == 'lit':
        model = LiT(**model_args)
    elif model_type.lower() == 'siglit':
        model = SigLiT(**model_args)
    elif model_type.lower() == 'crosslingual':
        model = CrossLingual(**model_args)
    else:
        model = mCLIP(**model_args)
    return model

def get_dataloader(train_args, model_args, train = True): 
    # Get lists of dataloader for each dataset
    datasets = train_args['dataset']
    image_folder = train_args['image_folder']
    training_objective = model_args['model_type']
    batch_size = train_args['batch_size']
    num_workers = train_args['num_workers']
    is_ddp = train_args['ddp']
    
    sampler = None
    dataloaders = []
    samplers = []
    
    if isinstance(datasets, str):
        datasets = [datasets]
    
    for data in datasets:
        df = pd.read_parquet(f'{data}.parquet')
        if training_objective in ['clip','siglip','lit','siglit']:
            dataset = ImageCaptionDataset(df, os.path.join(image_folder, data.split('/')[-1]))
            sampler = CLIPSampler(duplicate_id = 0, batch_size = batch_size)
        elif training_objective == 'crosslingual':
            
            dataset = CrossLingualDataset(df)
        else:
            dataset = mCLIPDataset(df, os.path.join(image_folder, data.split('/')[-1]))
        
        if is_ddp:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(dataset)
            
        if train and training_objective != 'crosslingual':
            dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = not is_ddp, sampler=sampler, num_workers = num_workers)
        else:
            dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = not is_ddp, num_workers = num_workers)
        
        dataloaders.append(dataloader)
        samplers.append(sampler)

    return dataloaders, samplers