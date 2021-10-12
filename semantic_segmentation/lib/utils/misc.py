from typing import List
import torch
from torchvision.transforms.functional import normalize
import numpy as np

def denormalize(tensor: torch.Tensor, mean: List, std: List):
    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean/std
    _std = 1/std
    return normalize(tensor, _mean, _std)
