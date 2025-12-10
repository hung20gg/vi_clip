
""" 
    Default arguments for training, model and evaluation
    
"""
training_args = {
    'train_name':'siglip',
    'wandb_project':'cv_final',
    
    'train_type':'single', # 'single', 'ddp' or 'dp'
    'mixed_precision': False,
    'device': 'cuda',
    'lr': 1e-4,
    'weight_decay': 1e-7,
    'epochs': 10,
    'batch_size': 2304,
    'scheduler': 'cosine', # 'cosine' or 'linear'
    'warmup_steps': 500,
    'peak_lr': 1,
    'initial_lr': 0.01,
    'num_workers': 16,
    'dataset': ['../data/cc3m-siglip-b224', '../data/cc12m-siglip-b224'], # Directory of the dataset
    'dataset_trim': 4,
    'data_type': 'numpy', # 'numpy' or 'images'
    'save_dir': 'checkpoints/text_model_base',
    'save_text_projection': 'checkpoints/text_projection_base',
    'train_projection_only' : False,
    'text_projection_lr': 2e-4,
    'text_projection_iters': 1000,
    'train_text': True,
    'accelerate': False,
    'evaluate_every': 1000,
    'log_every': 20,
    'hf_repo_name': 'hung20gg/vi_clip_v2',
    'beta2': 0.95 # On siglip, 0.95 is used. Else, 0.999
    
}

model_args = {
    'text_model': 'vinai/phobert-base-v2',
    'vision_model': 'vit_base_patch16_siglip_224',
    'checkpoint': None,
    'checkpoint_type': 'text', # 'text' or 'prj'
    'checkpoint_source': 'local', # 'local' or 'huggingface'
    'max_length': 64,
    'model_type': 'text_siglip', # 'text_siglip' or 'text_clip'
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
        'train_name': args.train_name,
        'wandb_project': args.wandb_project,
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
        'initial_lr': args.initial_lr,
        'num_workers': args.num_workers,
        'dataset': args.dataset,
        'save_dir': args.save_dir,
        'evaluate_every': args.evaluate_every,
        'beta2': args.beta2,
        'data_type': args.data_type,
        'train_projection_only': args.train_projection_only,
        'log_every': args.log_every,
        'hf_repo_name': args.hf_repo_name,
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
