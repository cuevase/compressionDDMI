import torch 
import math 

shared_seed = 42 # First specify the same seed

def generate_noise(seed, shape):
    generator = torch.Generator()
    generator.manual_seed(seed)

    return torch.randn(shape, generator = generator)

