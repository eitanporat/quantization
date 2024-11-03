import copy
from consts import MIN_RANGE, MAX_RANGE
import torch.nn as nn

def create_quantized_model(module, layer_type, **kwargs):
    module_copy = copy.deepcopy(module)

    def replace_linear_with_layer_type(submodule):
        for name, layer in submodule.named_children():
            if isinstance(layer, nn.Linear):
                setattr(submodule, name, layer_type(layer, **kwargs))
            else:
                replace_linear_with_layer_type(layer)

    replace_linear_with_layer_type(module_copy)

    return module_copy