from collections import defaultdict
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import torch
from imagenet.dataset import ImagenetDataset
import re
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from typing import List

def plot_error(approx, real, title="Relative error distribution", ax=None):
    relative_error = ((approx - real + 1e-10).abs() / (real + 1e-10).abs()).flatten()
    relative_error = relative_error[torch.abs(relative_error) > 1e-5]
    log_relative_error = torch.log10(relative_error)
    if ax is None:
        fig, ax = plt.subplots(1,1)
        
    ax.set_xlabel("Log of relative error")
    ax.set_title(title)
    ax.hist(log_relative_error.detach().cpu().numpy(), bins=100)
    
def load_subset(dataset: ImagenetDataset, num_samples: int, **kwargs):
    torch.random.seed()
    np.random.seed(1337)
    indices = np.random.choice(len(dataset), size=num_samples, replace=False)
    subset = Subset(dataset, indices)
    return DataLoader(subset, **kwargs)

def compute_accuracy(loader, model):
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            # Top-1 accuracy
            _, pred_top1 = outputs.topk(1, 1, True, True)
            pred_top1 = pred_top1.t()
            correct = pred_top1.eq(labels.view(1, -1))
            correct_top1 += correct.sum().item()
            # Top-5 accuracy
            _, pred_top5 = outputs.topk(5, 1, True, True)
            pred_top5 = pred_top5.t()
            correct = pred_top5.eq(labels.view(1, -1).expand_as(pred_top5))
            correct_top5 += correct.sum().item()
            total += labels.size(0)

    # Bayesian estimation with Beta prior (uniform prior corresponds to Beta(1,1))
    a_top1 = correct_top1
    b_top1 = total - correct_top1
    a_top5 = correct_top5
    b_top5 = total - correct_top5

    # Posterior parameters
    alpha_top1 = a_top1 + 1
    beta_top1 = b_top1 + 1
    alpha_top5 = a_top5 + 1
    beta_top5 = b_top5 + 1

    # Posterior means
    acc1 = alpha_top1 / (alpha_top1 + beta_top1)
    acc5 = alpha_top5 / (alpha_top5 + beta_top5)

    # Posterior variances
    var_acc1 = (alpha_top1 * beta_top1) / ((alpha_top1 + beta_top1) ** 2 * (alpha_top1 + beta_top1 + 1))
    var_acc5 = (alpha_top5 * beta_top5) / ((alpha_top5 + beta_top5) ** 2 * (alpha_top5 + beta_top5 + 1))

    # Standard deviations
    acc1_std = var_acc1 ** 0.5
    acc5_std = var_acc5 ** 0.5

    return acc1, acc1_std, acc5, acc5_std, total

def test_model(dataset, model: nn.Module, num_samples: int, batch_size: int):
    loader = load_subset(dataset, num_samples, batch_size=batch_size)
    acc1, acc1_std, acc5, acc5_std, total = compute_accuracy(loader, model)
    print(f"acc@1 on the sample batch: {acc1 * 100:.2f}%")
    print(f"acc@5 on the sample batch: {acc5 * 100:.2f}%")

    return acc1, acc1_std, acc5, acc5_std, total

def count_parameters_by_type(model):
    param_count_by_type = defaultdict(int)

    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_type = re.sub(r'(\d+)', 'X', name)
            param_count_by_type[layer_type] += param.numel()

    for layer_type, total_count in sorted(list(param_count_by_type.items()), key=lambda x: x[1], reverse=True):
        print(f"{layer_type}: {total_count / 1_000_000:.2f}M")
        
def create_noise(tensor, integer_ranges):
    max_val = torch.max(tensor, axis=-1).values.unsqueeze(-1)
    min_val = torch.min(tensor, axis=-1).values.unsqueeze(-1)
    val_range = max_val - min_val
    val_range = torch.maximum(val_range, torch.full_like(val_range, 1e-10)) # in the case of zeros
    noise = torch.randint(0, 2, tensor.shape).to(tensor.device, torch.float32)
    noise *= (1/(integer_ranges - 1)) * val_range
    return noise

def invert_matrices(matrices):
    matrices_shape = matrices.shape
    matrices = matrices.view(-1, matrices.shape[-2], matrices.shape[-1])
    pinv = torch.linalg.pinv(matrices)
    return pinv.view(matrices_shape)

def invert_permutations(permutations):
    return torch.argsort(permutations, axis=-1)

def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors

def permute_matrix_operation(a: torch.Tensor, permutations: torch.Tensor, matrices: List[torch.Tensor]):
    assert len(a.shape) in [2,3]
    n = a.shape[0] if len(a.shape) == 2 else a.shape[1]
    assert all(set(permutation.tolist()) == set(range(n)) for permutation in permutations)
    
    a_shape = a.shape

    if len(a.shape) == 2:
        a = a.unsqueeze(0)
    
    for permutation, matrix in zip(permutations, matrices):
        new_a = torch.zeros_like(a)
        group_size = matrix.shape[-1]
        num_groups = n // group_size
        grouped_a = a.reshape((a.shape[0], num_groups, group_size, a.shape[-1]))
        new_a = torch.einsum('gik,bgkj->bgij', matrix, grouped_a).reshape(a.shape)  
        a = new_a[:, permutation, :].clone()

    return a.reshape(a_shape)