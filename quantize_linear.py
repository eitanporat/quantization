import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import ortho_group
from consts import MIN_RANGE, MAX_RANGE
from quantized_matrix import QuantizedMatrix
from utils import *
from fast_hadamard_transform import hadamard_transform
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QuantizedLinear(nn.Module):
    def __init__(self, input_linear, min_range=MIN_RANGE, max_range=MAX_RANGE, show_plot=False):
        super(QuantizedLinear, self).__init__()
        self.in_features = input_linear.in_features
        self.out_features = input_linear.out_features
        self.min_range = min_range
        self.max_range = max_range
        self.weight = nn.Parameter(input_linear.weight.clone())
        self.qw = QuantizedMatrix(self.weight, self.min_range, self.max_range)
        
        self.show_plot = show_plot
        
        if self.show_plot:
            plot_error(self.qw.dequantize(), self.weight, title='Quantization error for weights of linear layer')

        if input_linear.bias is not None:
            self.bias = nn.Parameter(input_linear.bias.clone())
        else:
            self.bias = None


    def forward(self, x):
        dim_two = False
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            dim_two = True

        qx = QuantizedMatrix(x, self.min_range, self.max_range)

        if self.show_plot:
            fig, axes = plt.subplots(1,2)
            axes[0].hist(x.ravel().cpu().numpy(), bins=255)
            axes[0].set_title('unquantized input')        
            axes[1].hist(qx.q.ravel().cpu().numpy(), bins=255)
            axes[1].set_title('quantized input')
            plt.show()
            
        res = qx @ self.qw + self.bias if self.bias is not None else qx @ self.qw

        if dim_two:
            res = res.squeeze(0)
        return res

    def __repr__(self):
        return f"QuantizedLinear(in_features={self.in_features}, out_features={self.out_features})"

class QuantizedHadamardLinear(nn.Module):
    def __init__(self, input_linear, min_range=MIN_RANGE, max_range=MAX_RANGE, show_plot=False):
        super(QuantizedHadamardLinear, self).__init__()
        self.in_features = input_linear.in_features
        self.out_features = input_linear.out_features
        self.min_range = min_range
        self.max_range = max_range

        lg2 = math.ceil(math.log(self.in_features, 2))
        self.padding = 2 ** lg2 - self.in_features
        self.padded_dim = 2 ** lg2
        self.weight = nn.Parameter(input_linear.weight.clone())

        self.hadamard_weight = hadamard_transform(F.pad(input_linear.weight.clone(), (0, self.padding)))
        self.qw = QuantizedMatrix(self.hadamard_weight, self.min_range, self.max_range)

        if show_plot:
            plot_error(self.qw.dequantize(), self.hadamard_weight, title='Quantization error for weights of linear layer')

        if input_linear.bias is not None:
            self.bias = nn.Parameter(input_linear.bias.clone())
        else:
            self.bias = None


    def forward(self, x):
        dim_two = False
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            dim_two = True

        qx = QuantizedMatrix(hadamard_transform(F.pad(x, (0, self.padding))), self.min_range, self.max_range)
        res = qx @ self.qw / self.padded_dim + self.bias

        if dim_two:
            res = res.squeeze(0)
        return res

    def __repr__(self):
        return f"QuantizedHadamardLinear(in_features={self.in_features}, out_features={self.out_features})"
    
    
class QuantizedPermutationLinear(nn.Module):
    def __init__(self, input_linear, min_range=MIN_RANGE, max_range=MAX_RANGE, show_plot=False):
        super(QuantizedPermutationLinear, self).__init__()
        self.in_features = input_linear.in_features
        self.out_features = input_linear.out_features
        self.min_range = min_range
        self.max_range = max_range
        self.weight = nn.Parameter(input_linear.weight.clone())
        
        self.factors = prime_factors(self.in_features)
        self.rounds = len(self.factors)
        self.permutations = torch.stack([torch.randperm(self.in_features) for _ in range(self.rounds)]).to(device)
        self.matrices = [torch.from_numpy(ortho_group.rvs(size=self.in_features // factor, dim=factor)).to(device, torch.float32) for factor in self.factors]
        self.inverse_matrices_mT = [invert_matrices(matrix).mT.to(device) for matrix in self.matrices]   
        self.permuted_weight = permute_matrix_operation(self.weight.data.mT, self.permutations, self.inverse_matrices_mT).mT

        self.qw = QuantizedMatrix(self.permuted_weight, self.min_range, self.max_range)
        
        self.show_plot = show_plot
        
        if self.show_plot:
            plot_error(self.qw.dequantize(), self.permuted_weight, title='Quantization error for weights of linear layer')

        if input_linear.bias is not None:
            self.bias = nn.Parameter(input_linear.bias.clone())
        else:
            self.bias = None


    def forward(self, x):
        dim_two = False
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            dim_two = True

        permuted_x = permute_matrix_operation(x.mT, self.permutations, self.matrices).mT
        qx = QuantizedMatrix(permuted_x, self.min_range, self.max_range)

        if self.show_plot:
            fig, axes = plt.subplots(1,2)
            axes[0].hist(x.ravel().cpu().numpy(), bins=255)
            axes[0].set_title('unquantized input')        
            axes[1].hist(qx.q.ravel().cpu().numpy(), bins=255)
            axes[1].set_title('quantized input')
            plt.show()
            
        res = qx @ self.qw + self.bias if self.bias is not None else qx @ self.qw

        if dim_two:
            res = res.squeeze(0)
        return res

    def __repr__(self):
        return f"QuantizedPermutationLinear(in_features={self.in_features}, out_features={self.out_features})"

    
class QuantizedPermutationNoiseLinear(nn.Module):
    def __init__(self, input_linear, min_range=MIN_RANGE, max_range=MAX_RANGE, show_plot=False):
        super(QuantizedPermutationNoiseLinear, self).__init__()
        self.in_features = input_linear.in_features
        self.out_features = input_linear.out_features
        self.min_range = min_range
        self.max_range = max_range
        self.weight = nn.Parameter(input_linear.weight.clone())
        self.weight_noise = create_noise(self.weight, self.max_range - self.min_range + 1)
        self.factors = prime_factors(self.in_features)
        self.rounds = len(self.factors)
        self.permutations = torch.stack([torch.randperm(self.in_features) for _ in range(self.rounds)]).to(device)
        self.matrices = [torch.from_numpy(ortho_group.rvs(size=self.in_features // factor, dim=factor)).to(device, torch.float32) for factor in self.factors]
        self.inverse_matrices_mT = [invert_matrices(matrix).mT.to(device) for matrix in self.matrices]   
        self.permuted_weight = permute_matrix_operation((self.weight+self.weight_noise).mT, self.permutations, self.inverse_matrices_mT).mT
        self.qw = QuantizedMatrix(self.permuted_weight, self.min_range, self.max_range)
        self.show_plot = show_plot
        
        if self.show_plot:
            plot_error(self.qw.dequantize(), self.permuted_weight, title='Quantization error for weights of linear layer')

        if input_linear.bias is not None:
            self.bias = nn.Parameter(input_linear.bias.clone())
        else:
            self.bias = None


    def forward(self, x):
        dim_two = False
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            dim_two = True

        noise_x = create_noise(x, self.max_range - self.min_range + 1)
        permuted_x = permute_matrix_operation((x + noise_x).mT, self.permutations, self.matrices).mT
        qx = QuantizedMatrix(permuted_x, self.min_range, self.max_range)
        res = qx @ self.qw + self.bias if self.bias is not None else qx @ self.qw
        res -= x @ self.weight_noise.T
        res -= noise_x @ (self.weight + self.weight_noise).T
        if dim_two:
            res = res.squeeze(0)

        if self.show_plot:
            fig, axes = plt.subplots(2,2)
            axes[0,0].hist(x.ravel().cpu().numpy(), bins=255)
            axes[0,0].set_title('unquantized input')        
            axes[0,1].hist(qx.q.ravel().cpu().numpy(), bins=255)
            axes[0,1].set_title('quantized input')
            axes[1,0].hist((x+noise_x).ravel().cpu().numpy(), bins=255)
            axes[1,0].set_title('x + noise_x distribution')
            axes[1,1].hist((permuted_x).ravel().cpu().numpy(), bins=255)
            axes[1,1].set_title('permuted_x distribution')
            
            plot_error(res, x @ self.weight.T + self.bias, 'intermediate relative error for layers of neural network')
            plt.show()
            
        return res

    def __repr__(self):
        return f"QuantizedPermutationNoiseLinear(in_features={self.in_features}, out_features={self.out_features})"