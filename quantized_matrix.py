import torch
import torch.nn as nn
import numpy as np

MIN_RANGE = -128
MAX_RANGE = 127

def quantize(tensor, new_min=MIN_RANGE, new_max=MAX_RANGE, dtype=torch.float32):
    min_val = torch.min(tensor, axis=-1).values.unsqueeze(-1)
    max_val = torch.max(tensor, axis=-1).values.unsqueeze(-1)
    scale = (new_max - new_min) / (max_val - min_val)
    scale = torch.maximum(scale, torch.full_like(scale, 1e-10)) # in the case of zeros
    result = torch.round((tensor - min_val) * scale + new_min).to(dtype)
    bias = new_min - scale * min_val
    return result, scale[...,0], bias[...,0]

def int8_32_matmul(a, b):
    return torch.matmul(a,b)

def dequantize(tensor, scale, bias, dtype=torch.float32):
    return ((tensor.to(dtype) - bias.unsqueeze(-1)) / scale.unsqueeze(-1)).to(dtype)

class QuantizedMatrix:
    def __init__(self, matrix, min_range=MIN_RANGE, max_range=MAX_RANGE):
        self.matrix = matrix
        self.min_range = min_range
        self.max_range = max_range
        
        (self.q, self.s, self.b) = quantize(self.matrix, new_min=self.min_range, new_max=self.max_range)
        self.q_minus_b = (self.q - self.b.unsqueeze(-1)).sum(axis=-1)
        self.inv_s = 1./self.s
        
    def __matmul__(self, other) -> np.ndarray:
        z = int8_32_matmul(self.q, other.q.T)
        E_1 = torch.einsum('bi,j->bij', -self.b, other.q_minus_b)
        E_2 = torch.einsum('bi,j->bij', self.q.sum(axis=-1), -other.b)
        res = torch.einsum('bij,bi,j->bij', z + E_1 + E_2, self.inv_s, other.inv_s)
        return res
    
    def dequantize(self):
        return dequantize(self.q, self.s, self.b)
    
