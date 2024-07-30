from torch.optim.lr_scheduler import LambdaLR
import math

def linear_warmup_decay_scheduler(optimizer, warmup_steps, total_steps, initial_lr, peak_lr):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps)) * initial_lr
        else:
            # Linear decay
            return initial_lr + (peak_lr - initial_lr) * (float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps)))
    
    return LambdaLR(optimizer, lr_lambda)

def cosine_warmup_scheduler(optimizer, warmup_steps, total_steps, lr, eta_min = 0):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps)) * lr
        else:
            # Cosine annealing
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return eta_min + 0.5 * (lr - eta_min) * (1 + math.cos(math.pi * progress))
    
    return LambdaLR(optimizer, lr_lambda)