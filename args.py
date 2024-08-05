training_args = {
    'lr': 5e-4,
    'weight_decay': 1e-3,
    'epochs': 10,
    'batch_size': 2048,
    'scheduler': 'cosine',
    'warmup_steps': 1000,
    'peak_lr': 1e-3,
    'num_workers': -1,
    'dataset': ['dfn_20', 'image_caption', 'sharegpt4v','wit'],
    'save_dir': 'checkpoints',
    'evaluate_every': 100
}

model_args = {
    'text_model': 'vinai/phobert-base-v2',
    'vision_model': 'vit_base_patch16_siglip_224',
    'model_type': 'siglip'
}

eval_args = {
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
        'num_workers': args.num_workers,
        'dataset': args.dataset,
        'save_dir': 'checkpoints',
        'evaluate_every': 100
    }

    model_args = {
        'text_model': args.text_model,
        'vision_model': args.vision_model,
        'model_type': args.model_type
    }

    eval_args = {
        'batch_size': args.eval_batch_size,
        'num_workers': args.eval_num_workers,
        'dataset': args.eval_dataset,
    }
    return training_args, model_args, eval_args