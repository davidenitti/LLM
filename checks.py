import random
import torch

DISABLE = True

def check_tensors(x, name, disable=DISABLE, prob=0.00005, nan_inf_always=False, min_val_print=0):
    if disable:
        return
    do_check = random.random() < prob
    if do_check:
        max_val = x.detach().max().item()
        min_val = x.detach().min().item()
        mean = x.detach().mean().item()
        max_abs = max(abs(max_val),abs(min_val))
        if max_abs > min_val_print:
            print(f"  {name} min {min_val:6.3f} mean {mean:6.3f} max {max_val:6.3f}")
    if nan_inf_always or do_check:
        if torch.isinf(x).any():
            print(f"inf in tensor {name}")
            exit()
        if torch.isnan(x).any():
            print(f"nan in tensor {name}")
            exit()

def check_model(model, disable=DISABLE, prob=0.001):
    if disable:
        return
    if random.random()<prob:
        for n, p in model.named_parameters():
            max_val = p.detach().abs().max()
            if max_val > 3:
                print(n,"MAX VAL",max_val.detach().item())
