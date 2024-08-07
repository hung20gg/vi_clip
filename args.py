training_args = {
    'train_type':'single',
    'lr': 5e-4,
    'weight_decay': 1e-3,
    'epochs': 10,
    'batch_size': 2048,
    'scheduler': 'cosine',
    'warmup_steps': 500,
    'peak_lr': 1,
    'intial_lr': 0.01,
    'num_workers': -1,
    'dataset': ['data/dfn_20', 'data/image_caption', 'data/sharegpt4v','data/wit'],
    'image_folder': 'data/images',
    'save_dir': 'checkpoints/text_model_base',
    'evaluate_every': 200,
    'beta2': 0.999
}

model_args = {
    'text_model': 'vinai/phobert-base-v2',
    'vision_model': 'vit_base_patch16_siglip_224',
    'max_length': 64,
    'model_type': 'siglip'
}

eval_args = {
    'is_eval': True,
    'batch_size': 2048,
    'num_workers': -1,
    'dataset': ['imagenet1k'],
}

def parse_to_train_model_eval_args(args):
    training_args = {
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
        'model_type': args.model_type,
        'max_length': args.max_length,
    }

    eval_args = {
        'is_eval': args.is_eval,
        'batch_size': args.eval_batch_size,
        'num_workers': args.eval_num_workers,
        'dataset': args.eval_dataset,
    }
    return training_args, model_args, eval_args