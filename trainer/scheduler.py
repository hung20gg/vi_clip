from torch.optim.lr_scheduler import LambdaLR
import torch
import math

def linear_warmup_decay_scheduler(optimizer, warmup_steps, total_steps, initial_lr, peak_lr):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps)) * peak_lr
        else:
            # Linear decay
            return peak_lr - (peak_lr - initial_lr) * (float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps - 1)))
    
    return LambdaLR(optimizer, lr_lambda)

def cosine_warmup_scheduler(optimizer, warmup_steps, total_steps, lr, eta_min = 0):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps)) * lr
        else:
            # Cosine annealing
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps - 1))
            return eta_min + 0.5 * (lr - eta_min) * (1 + math.cos(math.pi * progress))
    
    return LambdaLR(optimizer, lr_lambda)

if __name__ == '__main__':
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.linear = torch.nn.Linear(10, 10)
            self.linear2 = torch.nn.Linear(10, 1)
        
        def forward(self, x):
            return self.linear(self.linear2(x))
        
    model = DummyModel()
    model2 = DummyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001)
    print(model.linear.parameters())
    scheduler = cosine_warmup_scheduler(optimizer, 5, 100, 1)
    print(optimizer.param_groups)
    for i in range(100):
        optimizer.zero_grad()
        print(scheduler.get_lr())
        optimizer.step()
        scheduler.step()