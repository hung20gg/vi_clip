from ..model import CLIP, SigLIP, LiT, SigLiT, CrossLingual, mCLIP, BaselineCLIP, TextEncoder, ProjectionHead
import os
import pandas as pd
from torch.utils.data import DataLoader
from .dataloader import ImageCaptionDataset, CLIPSampler, CrossLingualDataset, mCLIPDataset, TensorCaptionDataset, PreembedDataset
from torch.utils.data.distributed import DistributedSampler

def build_model(model_args):
    """Build model based on the model_args

    Args:
        model_args (dict): model arguments

    Returns:
        nn.Module: return the model based on the model_args. If not found, raise ValueError
    """
        
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
    elif model_type.lower() == 'mclip':
        model = mCLIP(**model_args)
    elif model_type.lower() == 'baseline':
        model = BaselineCLIP(**model_args)
    elif 'text' in model_type.lower():
        model = TextEncoder(**model_args)
    else:
        model = ProjectionHead(**model_args)
        
    if model_args.get('checkpoint', None) is not None:
        if ".pt" in model_args['checkpoint']:
            checkpoint = model_args['checkpoint']
            model.load_checkpoint(checkpoint, model_args['checkpoint_type'])
    
    return model

def get_dataloader(train_args, model_args, train = True, device = 'cuda'): 
    # Get lists of dataloader for each dataset
    """
    Create list of dataloaders and samplers for each dataset.

    Args:
        train_args (dict): args for training
        model_args (dict): args for model
        train (bool, optional): adding shuffle for training. Defaults to True.

    Returns:
        list[(Dataloader, Sampler)]: Return list of dataloader and sampler for each dataset.
            If the training is not distributed, the sampler will be None.
    """
    trim_pos = train_args.get('dataset_trim', 0)
    datasets = train_args['dataset']
    training_objective = model_args['model_type']
    batch_size = train_args['batch_size']
    num_workers = train_args['num_workers']
    is_ddp = train_args['train_type'] == 'ddp'
    is_text_seg = 'phobert' in model_args['text_model']
    
    sampler = None
    dataloaders = []
    samplers = []
    
    if isinstance(datasets, str):
        datasets = [datasets]
    if isinstance(trim_pos, int):
        trim_pos = [trim_pos] * len(datasets)
        
    assert len(datasets) == len(trim_pos), "Length of dataset and trim_pos should be the same"
    
    for data, trim in zip(datasets, trim_pos):
        df = None
        
        for file in os.listdir(data):
            if file.endswith('.parquet'):
                df = pd.read_parquet(os.path.join(data, file))
        assert df is not None, "No parquet file found in the directory"
        
        # Load the embeddings
        if model_args.get('data_type', 'images') == 'numpy' or train_args.get('data_type', 'images') == 'numpy':
            print("Loading numpy files")
            if model_args.get('model_type', "text_siglip").split('_')[0] == 'text':
                dataset = TensorCaptionDataset(df, os.path.join(data, 'numpy'), type_ = 'numpy', trim = trim, segment = is_text_seg)
            else:
                print("Loading pre-embedded text, it might take a while")
                dataset = PreembedDataset(df, os.path.join(data, 'numpy'), type_ = 'numpy', trim = trim, text_model_name = model_args['text_model'], device = device)
        
        else:
            print("Loading images")
            # Load the images
            if training_objective in ['clip','siglip','lit','siglit', 'text_clip', 'text_siglip']:
                dataset = ImageCaptionDataset(df, os.path.join(data, 'images'), trim = trim, segment = is_text_seg)
                # sampler = CLIPSampler(duplicate_id = 0, batch_size = batch_size)
            elif training_objective == 'crosslingual':
                
                dataset = CrossLingualDataset(df)
            else:
                dataset = mCLIPDataset(df, os.path.join(data, 'images'))
        
        if is_ddp:
            # from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(dataset)
            
        if train:
            dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = not is_ddp, sampler=sampler, num_workers = num_workers)
        else:
            dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers)
        
        dataloaders.append(dataloader)
        samplers.append(sampler)

    return dataloaders, samplers