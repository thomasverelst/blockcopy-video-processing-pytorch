from __future__ import annotations

import warnings
from collections import deque
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from ..utils.profiler import timings

from .block_funcs import CombineFunction, SplitFunction, TransferFunction
from .blockpad import pad

VERBOSE = False

def is_tensorwrapper(x) -> bool:
    """
    check if given object x is TensorWrapper datatype
    """
    return isinstance(x, TensorWrapper)

def is_block(x) -> bool:
    """
    check if given object x is TensorWrapper datatype and uses blocked representation
    """
    return isinstance(x, TensorWrapper) and x.is_blocks

def to_tensorwrapper(x : torch.Tensor) -> TensorWrapper:
    """
    converts tensor into TensorWrapper datatype
    """
    assert x.is_cuda, "x must be on CUDA device!"
    return x.as_subclass(TensorWrapper)

def to_tensor(x : TensorWrapper) -> torch.Tensor:
    """
    converts tensorwrapper or list of tensorwrappers to torch.Tensor
    """
    if isinstance(x, TensorWrapper):
        return x.to_tensor()
    if isinstance(x, list):
        return [to_tensor(z) for z in x]
    if isinstance(x, tuple):
        return tuple([to_tensor(z) for z in x])
    if isinstance(x, dict):
        return {k:to_tensor(v) for k, v in x.items()}
    return x

def combine(x, inplace=False):
    if isinstance(x, TensorWrapper):
        return x.block_combine(x, inplace=inplace).to_tensor()
    if isinstance(x, list):
        return [combine(z, inplace=inplace) for z in x]
    if isinstance(x, tuple):
        return tuple([combine(z, inplace=inplace) for z in x])
    if isinstance(x, dict):
        return {k:combine(v, inplace=inplace) for k, v in x.items()}
    return x     

OPS = {
    'PADDED': set(['conv2d', 'max_pool2d', 'avg_pool2d', 'lp_pool2d', 'fractional_max_pool2d']),

    # list of interpolate operations (might require workaround for speed with bilinear interpolation)
    'INTERPOLATE': set(['interpolate', 'upsample_bilinear']),

    # list of operations that depend on the batch size (uses a slow workaround)
    'BATCHED': set(['group_norm']),

    # incompatible operations
    'INCOMPATIBLE': set(['adaptive_avg_pool2d', 'adaptive_max_pool2d', 'linear','flip','unsqueeze','reshape','view']),

    # only support channel dimension
    'CHANNELONLY': set(['mean','sum','max','min,','std','var','argmax','count_nonzero','nonzero']),

    # might be incompatible
    'WARNING': set([''])
}
OPS_SPECIAL = set().union(*OPS.values())


@torch.jit.script
def process_grid(n_exec: int, grid: torch.Tensor, n_total: int, n_transfer: int, not_grid: torch.Tensor):
    idx_exec = torch.arange(n_exec, dtype=torch.int32, device='cpu') # id per executed block
    grid_idx = torch.empty(grid.shape, dtype=torch.int32, device='cpu')
    grid_idx = grid_idx.masked_scatter_(grid, idx_exec)
    mapping_exec = torch.nonzero(grid.flatten()).squeeze(1)
    idx_transfer = torch.arange(-n_total, -n_total+n_transfer, dtype=torch.int32, device='cpu') # id per tansferred block
    grid_idx = grid_idx.masked_scatter_(not_grid, idx_transfer)
    return grid_idx, mapping_exec

class BlockFeatures():
    '''
    FIFO feature stack for temporal propagation
    '''
    def __init__(self, device):
        self.device = device
        self._grid = None # binary grid indicating the blocks to be executed (1) or transfered (0)
        self._grid_idx = None 
        self._mapping_exec = None
        self._transfer_idx = None
        
        self._features_computed = deque() # features of executed blocks
        self._features_transfer = deque() # features of transfered blocks
        self._features_full = deque() # features after combination
        
    def _process_grid(self, grid : torch.Tensor, meta_prev: BlockFeatures = None) -> None:
        with timings.env('tensorwrapper/process_grid', 10):
            if VERBOSE:
                print('TensorWrapper >> PROCESS GRID')
            
            assert grid.dim() == 4 # N, 1, G_H, G_W
            assert grid.shape[1] == 1

            grid = grid.to('cpu', dtype=torch.bool)
            not_grid = torch.logical_not(grid)
            n_total = grid.numel()         # total amount of blocks
            n_exec = int(torch.count_nonzero(grid))       # executed blocks
            n_transfer = n_total - n_exec # transferred blocks
            
            if meta_prev is None:
                assert n_exec == n_total, "no previous features known, should execute all blocks!"
            
            grid_idx, mapping_exec = process_grid(n_exec, grid, n_total, n_transfer, not_grid)
            
            self._grid = grid.to(self.device, dtype=torch.bool)
            self._grid_idx = grid_idx.to(self.device, dtype=torch.int32)
            self._mapping_exec = mapping_exec.to(self.device, dtype=torch.int32)
            
            if meta_prev is not None:
                prev_grid_idx = meta_prev._grid_idx
                self._transfer_idx = prev_grid_idx[not_grid].to(self.device)
            
        
    def store_features(self, data_computed: TensorWrapper, data_transfer: TensorWrapper, padding: int = -1) -> None:
        """
        put features on stack
        """
        assert data_computed.shape[1:] == data_transfer.shape[1:], f"Number of channels must be equal, got {(data_computed.shape, data_transfer.shape)}"
        
        self._features_computed.append((data_computed.detach(), padding))
        self._features_transfer.append((data_transfer.detach(), padding))
        
    def get_features(self) -> Tuple[Tuple[TensorWrapper, int], Tuple[TensorWrapper, int]]:
        """
        pop features from the stack (FIFO)
        """
        if len(self._features_computed) == 0:
            raise AssertionError("No computed features to pop from stack, something seems wrong in the model.")
        computed = self._features_computed.popleft()
        if len(self._features_transfer) == 0:
            transfer = torch.empty((0,computed.size(1),computed.size(2),computed.size(3)), device=self.device, dtype=self.dtype)
        else:
            transfer = self._features_transfer.popleft()
        return computed, transfer
        
    def store_features_full(self, data: TensorWrapper):
        """
        Store combined features
        """
        self._features_full.append(data.detach())
        
    def get_features_full(self):
        """
        Pop combined features (FIFO)
        """
        if len(self._features_full) == 0:
            raise AssertionError
        out = self._features_full.popleft()
        return out
        
    def clear(self):
        """
        remove temporal features
        """
        self._features_computed.clear()
        self._features_transfer.clear()
        self._features_full.clear()

class TensorWrapper(torch.Tensor):    
    """
    General representation for block-sparse tensors with temporal feature propagation
    """
    is_init = False
    def __init_meta(self, other=None):
        if not self.is_init:
            if other is None:
                self._is_blocks = False
                self._features = None
                self._features_prev = None
            else:
                self._is_blocks = other._is_blocks
                self._features = other._features
                self._features_prev = other._features_prev
            self.is_init = True
        return self

    def process_temporal_features(self, features_prev: BlockFeatures = None) -> BlockFeatures:
        if not self.is_init:
            self.__init_meta()
        self._features_prev = features_prev
        self._features = BlockFeatures(device=self.device)
        return self._features
    
    @staticmethod
    def _round(number: int, total: int, fraction: float = 1/16) -> int:
        multiple = int(total*fraction)
        out = multiple * (1 + (number - 1) // multiple)
        return out
        
    @property
    def data_shape(self) -> torch.Size:
        return self.shape
    
    @property
    def is_blocks(self) -> bool:
        return self._is_blocks
    
    @property  
    def block_size(self) -> int:
        return self.data_shape[-1] if self.is_blocks else -1

    def get_grid(self) -> torch.BoolTensor:
        return self._features._grid
    
    def get_grid_idx(self) -> torch.IntTensor:
        return self._features._grid_idx
    
    def get_mapping_exec(self) -> torch.IntTensor:
        return self._features._mapping_exec
    
    def get_features(self) -> BlockFeatures:
        return self._features
    
    def to_blocks(self, grid : torch.BoolTensor) -> TensorWrapper:
        """ 
        Convert normal tensor representation to blocked tensor representation
        according to given grid
        """
        assert not self.is_blocks
        self._features._process_grid(grid, self._features_prev)
        
        assert grid.dim() == 4
        assert self.dim() == 4
        assert self.shape[2] % grid.shape[2] == 0
        assert self.shape[3] % grid.shape[3] == 0
        
        block_size = self.shape[2]//grid.shape[2]
        
        return self._split(block_size)
    
    def to_blocks_like(self, other : TensorWrapper) -> TensorWrapper:
        """ 
        Convert normal tensor representation to blocked tensor representation
        with same grid as other
        """
        self.__init_meta(other)
        self._is_blocks = False
        block_size = self.shape[2]//self.get_grid().shape[2]
        return self._split(block_size)
    
    
    def _split(self, block_size: int) -> TensorWrapper:
        assert self.is_init, 'need to call process_temporal_features before splitting in blocks!'
        with timings.env('tensorwrapper/split', 10):
            if VERBOSE:
                print('TensorWrapper >> BLOCK SPLIT')
            
            if self.is_blocks:
                raise AttributeError('TensorWrapper: already split in blocks!')
            
            if self.dim() != 4:
                raise AttributeError('TensorWrapper only supports 4D NCHW tensors!')
            
            N,C,H,W = self.shape
            if H % block_size != 0 or W % block_size != 0:
                raise AttributeError(f'TensorWrapper: Shape ({self.shape}) not divisibile by given block size ({block_size})!')
            
            grid_idx = self.get_grid_idx()
            mapping_exec = self.get_mapping_exec()
            _, _, GH, GW = grid_idx.shape
            
            block_size = W//GW
        
            if VERBOSE:
                print(f'TensorWrapper >> BLOCK SPLIT >> block size {block_size} with grid shape {(GH, GW)}')
            
            n_exec = len(mapping_exec)
            n_total = grid_idx.numel()
            size = (self._round(n_exec, n_total), C, block_size, block_size) if n_exec > 0 else (0,C,block_size,block_size)
            out = torch.empty(size, dtype=self.dtype, device=self.device)
            out = SplitFunction.apply(out, self, mapping_exec, grid_idx)
            out = out.as_subclass(TensorWrapper)
            out.__init_meta(self)
            out._is_blocks = True
            return out
    
    def to_tensor(self) -> torch.Tensor:
        """
        convert self to torch.Tensor
        combines blocks if needed
        """
        out = self._combine() if self.is_blocks else self
        return out.as_subclass(torch.Tensor)

    def block_combine(self) -> TensorWrapper:
        """
        combines blocks into full tensor
        returns TensorWrapper (with _is_blocks = False)
        """
        return self._combine(inplace=False)

    def block_combine_(self) -> TensorWrapper:
        """
        inplace version of self.block_combine
        """
        return self._combine(inplace=True)

    def _combine(self, inplace: bool = False) -> TensorWrapper:
        with timings.env('tensorwrapper/combine', 4):
            if VERBOSE:
                print('TensorWrapper >> BLOCK COMBINE')
            
            if not self.is_blocks:
                raise AttributeError('TensorWrapper: Not split in blocks!')
            
            grid_idx = self.get_grid_idx()
            mapping_exec= self.get_mapping_exec()
            
            _, C, BS, _ = self.shape
            N, _, GH, GW = grid_idx.shape
            H, W = GH*BS, GW*BS

            out_shape = (N, C, H, W)
            
            if len(mapping_exec) == grid_idx.numel():
                # all blocks executed, nothing transferred
                if self._features_prev:
                    _ = self._features_prev.get_features_full()
                out = torch.empty(out_shape, dtype=self.dtype, device=self.device)
            else:
                out = self._features_prev.get_features_full()
                if not inplace:
                    out = out.clone()
                assert out_shape == out.shape, (out_shape, out.shape)
            out = CombineFunction.apply(self.data, out, grid_idx, mapping_exec)
            out = out.as_subclass(TensorWrapper)
            out.__init_meta(self)
            out._is_blocks = False
                 
            self._features.store_features_full(out)
            return out
    
    def _transfer_from_prev(self, data):
        with timings.env('tensorwrapper/transfer', 10):
            N, C, H, W = data.shape 

            if self._features_prev is None:
                return torch.empty((0, C, H, W), dtype=data.dtype, device=self.device)
                
            (prev_computed, prev_padding), (prev_transfer, _) = self._features_prev.get_features()
            
            assert prev_computed.shape[1:] == (C, H, W)
            assert prev_transfer.shape[1:] == (C, H, W)
            
            prev_grid_idx = self._features_prev._grid_idx
            transfer_idx = self._features._transfer_idx
            shape = (len(transfer_idx), C, H, W)
            data_transfer = torch.empty(shape, dtype=prev_transfer.dtype, device=self.device)
            data_transfer = TransferFunction.apply(data_transfer, prev_computed, prev_transfer, prev_grid_idx, transfer_idx, prev_padding)
            del prev_computed, prev_transfer
            return data_transfer
    
    def __torch_function__(self, func: Callable, types: Tuple, args: Tuple = (), kwargs: Optional[Dict] = None) -> Any:
        """
        wraps every torch operation and modifies operation for blocked execution if needed
        """
        assert self.is_init
        if not self.is_blocks:
            # if not in blocks, just apply operation as normal
            out = super().__torch_function__(func, types, args, kwargs)
        
        if kwargs is None:
            kwargs = {}
            
        op = func.__name__
        if op in OPS_SPECIAL:
            if op in OPS['PADDED']:
                out = self._func_replace_paddding(func, types, args, kwargs)
            elif op in OPS['INTERPOLATE']:
                out = self._func_interpolate(func, types, args, kwargs)
            elif op in OPS['BATCHED']:
                out = self._func_batched(func, types, args, kwargs)
            elif op in OPS['CHANNELONLY']:
                if 'dim' in kwargs and kwargs['dim'] == 1:
                    pass
                else:
                    print(f'Operation {op} might behave differently with TensorWrapper when dim != 1!')
                out = super().__torch_function__(func, types, args, kwargs)
            elif op in OPS['WARNING']:
                warnings.warn(f'Operation {op} might behave differently with TensorWrapper!')
                out = super().__torch_function__(func, types, args, kwargs)
            elif op in OPS['INCOMPATIBLE']:
                raise AttributeError(f'Operation {op} not supported for TensorWrapper!')
            else:
                raise AttributeError
        else:
            out = super().__torch_function__(func, types, args, kwargs)
        
        try:
            # if TensorWrapper, initialize it
            out.__init_meta(self)
        except:
            pass
        return out
        
    def _func_replace_paddding(self, func, types, args, kwargs):
        args = list(args)
        padding = kwargs.get('padding', None)
        if padding is None:
            padding = args[4] if len(args) > 4 else 0
            
        zeros = 0
        if isinstance(padding, (tuple, list)):
            zeros = (0, 0)
            if padding[0] != padding[1]:
                raise NotImplementedError(f'Only support equal paddings, got {padding}')
            padding = padding[0]
        
        if padding > 0:
            data = args[0]
            data_transfer = self._transfer_from_prev(data)
            self._features.store_features(data, data_transfer, padding)
            
            grid_idx = self.get_grid_idx()
            mapping_exec = self.get_mapping_exec()
            with timings.env('tensorwrapper/pad', 10):
                args[0] = pad(data, data_transfer, grid_idx, mapping_exec, padding) # manually pad
        
            if 'padding' in kwargs:
                kwargs['padding'] = zeros
            else:
                args[4] = zeros
        
            with timings.env('tensorwrapper/pad_func', 11):
                ret = super().__torch_function__(func, types, args, kwargs)
        else:
            with timings.env('tensorwrapper/pad_func0', 11):
                ret = super().__torch_function__(func, types, args, kwargs)
        return ret
    
    def _func_interpolate(self, func, types, args, kwargs):
        """
        pytorch bilinear interpolation is very slow small tensor sizes.
        Curiously, trilinear performs correctly (tested in Pytorch 1.9),
        therefore replace bilinear with trilinear with one extra dummy dimension 
        """
        if kwargs['mode'] == 'bilinear':
            args = list(args)
            data = args[0]
            kwargs['mode'] = 'trilinear'
            if kwargs['scale_factor'] is not None:
                sf = kwargs['scale_factor']
                kwargs['scale_factor'] = (1, sf, sf)
            elif kwargs['size'] is not None:
                size = kwargs['size']
                kwargs['size'] = (data.shape[1], size[0], size[1])
            args[0] = data.as_subclass(torch.Tensor).unsqueeze(0)
            ret = func(*args, **kwargs)
            ret = ret.squeeze(0).as_subclass(TensorWrapper)
        else:
            ret = super().__torch_function__(func, types, args, kwargs)
        return ret
    
    def _func_batched(self, func, types, args, kwargs):
        """
        applies group norm on blocked representation
        """
        args = list(args)
        data = args[0].as_subclass(torch.Tensor)
        in_shape = data.shape
        data = data.permute(1,0,2,3)
        data = data.reshape(data.shape[0], -1).unsqueeze(0)
        args[0] = data
        types = list(types)
        types[0] = type(data)
        out =  super().__torch_function__(func, types, args, kwargs)
        out = out.as_subclass(torch.Tensor)
        out = out.squeeze(0).view(in_shape[1], in_shape[0], in_shape[2], in_shape[3]).permute(1,0,2,3).contiguous()
        out = out.as_subclass(TensorWrapper)
        return out
