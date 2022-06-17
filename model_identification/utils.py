import torch

def torch_random_tensor(min, max, size):
    return (max-min) * torch.rand(size=size) + min