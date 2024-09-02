
""" 
    Default arguments for training, model and evaluation
    
"""
training_args = {
    'train_name':'test',
    'wandb_project':'test vi_clip',
    
    'train_type':'single', # 'single', 'ddp' or 'dp'
    'mixed_precision': False,
    'device': 'cuda',
    'lr': 1e-4,
    'weight_decay': 1e-3,
    'epochs': 10,
    'batch_size': 2048,
    'scheduler': 'cosine', # 'cosine' or 'linear'
    'warmup_steps': 500,
    'peak_lr': 1,
    'intial_lr': 0.01,
    'num_workers': 8,
    'epoch_on_first_dataset': 10, # Predownload the dataset for the first epoch
    'dataset': ['data/dfn_20', 'data/image_caption', 'data/sharegpt4v','data/wit'], # Directory of the dataset
    'dataset_trim': [4,4,4,4],
    'image_folder': 'data/images', # Prefix for image folder (ignore for now)
    'data_type': 'numpy', # 'numpy' or 'images'
    'save_dir': 'checkpoints/text_model_base',
    'save_text_projection': 'checkpoints/text_projection_base',
    'train_projection_only' : True,
    'text_projection_lr': 5e-4,
    'evaluate_every': 200,
    'text_projection_iters': 1000,
    'train_text': True,
    'beta2': 0.999 # On siglip, 0.95 is used
    
}

model_args = {
    'text_model': 'vinai/phobert-base-v2',
    'vision_model': 'vit_base_patch16_siglip_224',
    'clip_model': 'google/siglip-base-patch16-224',
    'checkpoint': None,
    'max_length': 64,
    'model_type': 'siglip', # 'text_siglip' or 'text_clip'
    'pretrain': True,
    'projection_dim':768,
    'force_text_projection': True
}

eval_args = {
    'is_eval': True,
    'batch_size': 2048,
    'num_workers': 6,
    'dataset': 'imagenet1k',
}

def parse_to_train_model_eval_args(args):
    training_args = {
        'train_type': args.train_type,
        'mixed_precision': args.mixed_precision,
        'device': args.device,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'scheduler': args.scheduler,
        'warmup_steps': args.warmup_steps,
        'peak_lr': args.peak_lr,
        'intial_lr': args.intial_lr,
        'num_workers': args.num_workers,
        'dataset': args.dataset,
        'image_folder': args.image_folder,
        'save_dir': args.save_dir,
        'evaluate_every': 200,
        'beta2': args.beta2
    }

    model_args = {
        'text_model': args.text_model,
        'vision_model': args.vision_model,
        'clip_model': args.clip_model,
        'model_type': args.model_type,
        'max_length': args.max_length,
        'pretrain': args.pretrain,
        'force_text_projection': args.force_text_projection
        
    }

    eval_args = {
        'is_eval': args.is_eval,
        'batch_size': args.eval_batch_size,
        'num_workers': args.eval_num_workers,
        'dataset': args.eval_dataset,
    }
    return training_args, model_args, eval_args