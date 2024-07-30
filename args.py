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
}

model_args = {
    'text_encoder': 'vinai/phobert-base-v2',
    'image_encoder': 'vit_base_patch16_siglip_224',
    'model_type': 'siglip'
}

eval_args = {
    'batch_size': 2048,
    'num_workers': -1,
    'dataset': 'imagenet1k',
}