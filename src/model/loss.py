import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_function(q, k, queue):
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    return 2 - 2 * (x * y).sum(dim=-1)