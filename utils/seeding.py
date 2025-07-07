import random
import numpy as np
import torch
import hashlib

def seed_everything(seed):
    """Set random seeds for Python, NumPy, and PyTorch based on a string or int."""
    # Convert string to an integer hash
    if isinstance(seed, str):
        seed = int(hashlib.md5(seed.encode()).hexdigest(), 16) % (2**32) # 32-bit seed
    
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True

