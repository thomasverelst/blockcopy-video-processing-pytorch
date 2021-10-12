from sys import modules
import torch
import torch.nn as nn
from torch.nn.modules import module

def fuse_bn_sequential(module: nn.Module, module_name : str):
    """
    This function takes a sequential block and fuses the batch normalization with convolution

    :param model: nn.Sequential. Source resnet model
    :return: nn.Sequential. Converted block
    """
    children = list(module.children())
    i = 0
    for name, m in module.named_children():
        if i > 0 and isinstance(m, nn.BatchNorm2d):
            if hasattr(m, 'fused'):
                continue
            if m.training:
                continue
            if isinstance(children[i-1], nn.Conv2d):
                if hasattr(children[i-1], 'fused'):
                    continue
                bn_st_dict = m.state_dict()
                conv_st_dict = children[i-1].state_dict()

                # BatchNorm params
                eps = m.eps
                mu = bn_st_dict['running_mean']
                var = bn_st_dict['running_var']
                gamma = bn_st_dict['weight']

                if 'bias' in bn_st_dict:
                    beta = bn_st_dict['bias']
                else:
                    beta = torch.zeros(gamma.size(0)).float().to(gamma.device)

                # Conv params
                W = conv_st_dict['weight']
                if 'bias' in conv_st_dict:
                    bias = conv_st_dict['bias']
                else:
                    bias = torch.zeros(W.size(0)).float().to(gamma.device)
                try:

                    denom = torch.sqrt(var + eps)
                    b = beta - gamma.mul(mu).div(denom)
                    A = gamma.div(denom)
                    bias *= A
                    A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)

                    W.mul_(A)
                    bias.add_(b)

                    children[i-1].weight.data.copy_(W)
                    if children[i-1].bias is None:
                        children[i-1].bias = torch.nn.Parameter(bias)
                    else:
                        children[i-1].bias.data.copy_(bias)
                    
                    children[i-1].fused = True
                    children[i].fused = True
                    setattr(module, name, nn.Identity())
                    print(f'BatchNorm fused with Conv: {module_name}.{name}')
                except:
                    pass
        i += 1 


def fuse_bn_recursively(model):
    for m_name, m in model.named_modules():
        if len(list(m.children())) > 0:
            fuse_bn_sequential(m, m_name)

    return model