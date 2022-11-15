from __future__ import annotations

import warnings
from collections import deque
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from ..utils.profiler import timings

from ..utils.block_funcs import CombineFunction, SplitFunction, TransferFunction
from ..utils.blockpad import pad

VERBOSE = False # debugging flag to print verbose statements
BLOCKPAD_WITH_ZEROES = False # debugging flag to use zero-padding instead of block-padding


def is_tensorwrapper(x) -> bool:
    """
    Check if given object x is TensorWrapper datatype
    """
    return isinstance(x, TensorWrapper)


def is_block(x) -> bool:
    """
    Check if given object x is TensorWrapper datatype and uses blocked representation
    """
    return isinstance(x, TensorWrapper) and x.is_blocks


def to_tensorwrapper(x: torch.Tensor) -> TensorWrapper:
    """
    Converts tensor into TensorWrapper datatype
    """
    assert x.is_cuda, "x must be on CUDA device!"
    return x.as_subclass(TensorWrapper)


def to_tensor(x: TensorWrapper) -> torch.Tensor:
    """
    Converts TensorWrapper object or list of TensorWrappers to torch.Tensor
    """
    if isinstance(x, TensorWrapper):
        return x.to_tensor()
    if isinstance(x, list):
        return [to_tensor(z) for z in x]
    if isinstance(x, tuple):
        return tuple([to_tensor(z) for z in x])
    if isinstance(x, dict):
        return {k: to_tensor(v) for k, v in x.items()}
    return x


# def combine(x: TensorWrapper, inplace: bool = False) -> torch.Tensor:
#     """
#     # TODO remove
#     """
#     if isinstance(x, TensorWrapper):
#         return x.combine(x, inplace=inplace).to_tensor()
#     if isinstance(x, list):
#         return [combine(z, inplace=inplace) for z in x]
#     if isinstance(x, tuple):
#         return tuple([combine(z, inplace=inplace) for z in x])
#     if isinstance(x, dict):
#         return {k: combine(v, inplace=inplace) for k, v in x.items()}
#     return x


OPS = {
    # list of operations having padding - requires filling in zero-paddign with propagated features from the previousf rame
    "PADDED": set(["conv2d", "max_pool2d", "avg_pool2d", "lp_pool2d", "fractional_max_pool2d"]),
    # list of interpolate operations (might require workaround for speed with bilinear interpolation)
    "INTERPOLATE": set(["interpolate", "upsample_bilinear"]),
    # list of operations that depend on the batch size (uses a slow workaround)
    "BATCHED": set(["group_norm"]),
    # incompatible with block-wise execution  because they require values of the whole tensor
    "INCOMPATIBLE": set(
        [
            "adaptive_avg_pool2d",
            "adaptive_max_pool2d",
            "linear",
            "flip",
            "unsqueeze",
            "reshape",
            "view",
        ]
    ),
    # only support channel dimension
    "CHANNELONLY": set(
        [
            "mean",
            "sum",
            "max",
            "min,",
            "std",
            "var",
            "argmax",
            "count_nonzero",
            "nonzero",
        ]
    ),
    # might be incompatible
    "WARNING": set([""]),
}
OPS_SPECIAL = set().union(*OPS.values())  # set of all special operations listed in OPS


@torch.jit.script
def get_grid_mappings(
    n_exec: int,
    grid: torch.BoolTensor,
    not_grid: torch.BoolTensor,
    n_total: int,
    n_transfer: int,
):
    """
    Get mappings for other operations.

    """
    idx_exec = torch.arange(n_exec, dtype=torch.int32, device="cpu")  # id per executed block
    grid_idx = torch.empty(grid.shape, dtype=torch.int32, device="cpu")
    grid_idx = grid_idx.masked_scatter_(grid, idx_exec)
    mapping_exec = torch.nonzero(grid.flatten()).squeeze(1)
    idx_transfer = torch.arange(
        -n_total, -n_total + n_transfer, dtype=torch.int32, device="cpu"
    )  # id per tansferred block
    grid_idx = grid_idx.masked_scatter_(not_grid, idx_transfer)
    return grid_idx, mapping_exec


class BlockFeatures:
    """
    FIFO feature stack for temporal propagation of features
    for all intermediate features in the network
    """

    def __init__(self, device):
        self.device = device
        self._grid = None  # binary grid indicating the blocks to be executed (1) or transfered (0), 4-dim tensor
        self._grid_idx = None  # grid (same shape as self._grid) where each element indicates the cumulative sum of the number of blocks to be executed (for which self._grid == 1)
        self._mapping_exec = (
            None  # vector of flattened indices of the blocks to be executed (for which self._grid == 1)
        )
        self._transfer_idx = None  # vector of flattened indices of the blocks of the previous frame to be transfered

        self._features_computed = deque()  # features of executed blocks - for all layers in the network
        self._features_transfer = deque()  # features of transfered blocks - for all layers in the network
        self._features_full = deque()  # features after combination

    def _process_grid(self, grid: torch.Tensor, meta_prev: BlockFeatures = None) -> None:
        with timings.env("tensorwrapper/process_grid", 10):
            if VERBOSE:
                print("TensorWrapper >> PROCESS GRID")

            assert grid.dim() == 4  # N, 1, G_H, G_W
            assert grid.shape[1] == 1

            grid = grid.to("cpu", dtype=torch.bool)
            not_grid = torch.logical_not(grid)
            n_total = grid.numel()  # total number of blocks in batch
            n_exec = int(torch.count_nonzero(grid))  # number of executed blocks in batch
            n_transfer = n_total - n_exec  # number of transferred blocks in batch

            if meta_prev is None:
                assert n_exec == n_total, "No previous features known, first run should execute all blocks!"

            # get metadata from grid (mappings etc)
            grid_idx, mapping_exec = get_grid_mappings(n_exec, grid, not_grid, n_total, n_transfer)

            # move medata to GPU
            self._grid = grid.to(self.device, dtype=torch.bool)
            self._grid_idx = grid_idx.to(self.device, dtype=torch.int32)
            self._mapping_exec = mapping_exec.to(self.device, dtype=torch.int32)

            # find the indices of the blocks of the previous frame that should be transferred
            if meta_prev is not None:
                prev_grid_idx = meta_prev._grid_idx
                self._transfer_idx = prev_grid_idx[not_grid].to(self.device)

    def store_features(self, data_computed: TensorWrapper, data_transfer: TensorWrapper, padding: int = -1) -> None:
        """
        Add features on the stack
        """
        assert (
            data_computed.shape[1:] == data_transfer.shape[1:]
        ), f"Number of channels must be equal, got {(data_computed.shape, data_transfer.shape)}"

        self._features_computed.append((data_computed.detach(), padding))
        self._features_transfer.append((data_transfer.detach(), padding))

    def get_features(
        self,
    ) -> Tuple[Tuple[TensorWrapper, int], Tuple[TensorWrapper, int]]:
        """
        Pop features from the stack (FIFO)
        # TODO
        """
        if len(self._features_computed) == 0:
            raise AssertionError("No computed features to pop from stack, something seems wrong in the model.")
        computed = self._features_computed.popleft()
        if len(self._features_transfer) == 0:
            transfer = torch.empty(
                (0, computed.size(1), computed.size(2), computed.size(3)),
                device=self.device,
                dtype=self.dtype,
            )
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
        Remove all stord features
        """
        self._features_computed.clear()
        self._features_transfer.clear()
        self._features_full.clear()


class TensorWrapper(torch.Tensor):
    """
    General representation for block-sparse tensors with temporal feature propagation
    """

    is_init = False  # keep track if the instance has been initialized

    def __init_metadata(self, other=None):
        """
        initialize meta data
        """
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
        self.__init_metadata()
        self._features_prev = features_prev
        self._features = BlockFeatures(device=self.device)
        return self._features

    @property
    def data_shape(self) -> torch.Size:
        """
        Return the shape of the actual data
        """
        return self.shape

    @property
    def is_blocks(self) -> bool:
        """
        Return if the tensor is block-sparse
        """
        return self._is_blocks

    @property
    def block_size(self) -> int:
        """
        Return the block size in pixels.
        Returns -1 if not in blocks
        """
        return self.data_shape[-1] if self.is_blocks else -1

    def get_grid(self) -> torch.BoolTensor:
        """
        Return the bool grid indiciating the executed blocks
        """
        return self._features._grid

    def get_grid_idx(self) -> torch.IntTensor:
        return self._features._grid_idx

    def get_mapping_exec(self) -> torch.IntTensor:
        return self._features._mapping_exec

    def get_features(self) -> BlockFeatures:
        """
        Returns the BlockFeatures storing the recorded intermediate features
        """
        return self._features

    def to_blocks(self, grid: torch.BoolTensor) -> TensorWrapper:
        """
        Convert normal tensor representation to blocked tensor representation
        according to given grid

        Arguments:
            grid: 4-dimensional torch.BoolTensor with shape (N, 1, G_H, G_W)
            where N is the batch size, and G_H and G_W is the height/width of the block grid
        """
        assert not self.is_blocks
        self._features._process_grid(grid, self._features_prev)

        assert grid.dim() == 4
        assert self.dim() == 4
        assert self.shape[2] % grid.shape[2] == 0
        assert self.shape[3] % grid.shape[3] == 0

        block_size = self.shape[2] // grid.shape[2]

        return self._split(block_size)

    def to_blocks_like(self, other: TensorWrapper) -> TensorWrapper:
        """
        Convert normal tensor representation to blocked tensor representation
        with same grid as other
        """
        self.__init_metadata(other)
        self._is_blocks = False
        block_size = self.shape[2] // self.get_grid().shape[2]
        return self._split(block_size)

    def _split(self, block_size: int) -> TensorWrapper:
        """
        Split this tensor instance into blocks with given block size in pixels
        """
        assert self.is_init, "need to call process_temporal_features before splitting in blocks!"
        with timings.env("tensorwrapper/split", 10):
            if VERBOSE:
                print("TensorWrapper >> BLOCK SPLIT")

            if self.is_blocks:
                raise AttributeError("TensorWrapper: already split in blocks! Cannot split again.")

            if self.dim() != 4:
                raise AttributeError("TensorWrapper only supports 4D NCHW tensors!")

            N, C, H, W = self.shape
            if H % block_size != 0 or W % block_size != 0:
                raise AttributeError(
                    f"TensorWrapper: Shape ({self.shape}) not divisibile by given block size ({block_size})!"
                )

            grid_idx = self.get_grid_idx()
            mapping_exec = self.get_mapping_exec()
            _, _, GH, GW = grid_idx.shape

            block_size = W // GW  # block size in pixels

            if VERBOSE:
                print(f"TensorWrapper >> BLOCK SPLIT >> block size {block_size} with grid shape {(GH, GW)}")

            n_exec = len(mapping_exec)  # the number of executed blocks

            # size of the block-sparse representation
            size = (n_exec, C, block_size, block_size) if n_exec > 0 else (0, C, block_size, block_size)

            # allocate memory for the block-sparse representation
            out = torch.empty(size, dtype=self.dtype, device=self.device)

            # copy values for executed blocks from dense self into block-sparse out
            out = SplitFunction.apply(out, self.data, mapping_exec, grid_idx)

            # output is not a normal tensor, but subclass of TensorWrapper to record metadata
            out = out.as_subclass(TensorWrapper)
            out.__init_metadata(self)
            out._is_blocks = True

            return out

    def to_tensor(self) -> torch.Tensor:
        """
        Convert block-sparse data of self to torch.Tensor
        Combines the blocks into a dense tensor if needed
        """
        out = self.combine() if self.is_blocks else self
        return out.as_subclass(torch.Tensor)

    def combine_(self) -> TensorWrapper:
        """
        inplace version of self.combine()
        """
        return self.combine(inplace=True)

    def combine(self, inplace: bool = False) -> TensorWrapper:
        """
        Combine block-sparse self into dense tensor
        Output values for non-executed are copied from previous frame
        If inplace is True, the values are copied into the previous output tensor (slightly faster)
        If inplace is False, the previous output tensor is cloned and the vaues are then copied
        """
        with timings.env("tensorwrapper/combine", 4):
            if VERBOSE:
                print("TensorWrapper >> BLOCK COMBINE")

            if not self.is_blocks:
                raise AttributeError("TensorWrapper: Not split in blocks!")

            grid_idx = self.get_grid_idx()
            mapping_exec = self.get_mapping_exec()

            _, C, BS, _ = self.shape
            N, _, GH, GW = grid_idx.shape
            H, W = GH * BS, GW * BS

            out_shape = (N, C, H, W)

            # allocate memory for the dense representation
            if self._features_prev:
                # if there was a previously executed frame
                out = self._features_prev.get_features_full()
                if not inplace:
                    out = out.clone()
                assert out_shape == out.shape, (out_shape, out.shape)
            else:
                # if current frame is first frame of clip, all blocks were executed
                assert len(mapping_exec) == grid_idx.numel()
                out = torch.empty(out_shape, dtype=self.dtype, device=self.device)
                

            # copy data of block-sparse tensor to dense tensor based on mappings
            out = CombineFunction.apply(self.data, out, grid_idx, mapping_exec)

            # output is again a TensorWrapper object to record metadata
            out = out.as_subclass(TensorWrapper)
            out.__init_metadata(self)
            out._is_blocks = False

            # store the result of this combine, so that next frame can use it
            self._features.store_features_full(out)
            return out

    def _transfer_from_prev(self):
        """
        Get features from previous frame for blocks that should be transferred 
        Returns tensor of shape (NUM_BLOCKS_TRANSFERRED, C, H, W)
        If no previous frame, return None
        """
        with timings.env("tensorwrapper/transfer", 10):
            if self._features_prev is None:
                return None

            (prev_computed, prev_padding), (
                prev_transfer,
                _,
            ) = self._features_prev.get_features()

            assert prev_transfer.shape[1:] == prev_computed.shape[1:]
            _, C, H, W = prev_transfer.shape

            prev_grid_idx = self._features_prev._grid_idx
            transfer_idx = self._features._transfer_idx
            shape = (len(transfer_idx), C, H, W)
            data_transfer = torch.empty(shape, dtype=prev_transfer.dtype, device=self.device)
            data_transfer = TransferFunction.apply(
                data_transfer,
                prev_computed,
                prev_transfer,
                prev_grid_idx,
                transfer_idx,
                prev_padding,
            )
            del prev_computed, prev_transfer
            return data_transfer

    def __torch_function__(
        self,
        func: Callable,
        types: Tuple,
        args: Tuple = (),
        kwargs: Optional[Dict] = None,
    ) -> Any:
        """
        Intercepts every torch operation and modifies operation for block-sparse execution if needed
        """
        assert self.is_init
        
        # if not in blocks, just apply operation as normal
        if not self.is_blocks:
            out = super().__torch_function__(func, types, args, kwargs)

        if kwargs is None:
            kwargs = {}

        # handle the operation
        op = func.__name__
        if op in OPS_SPECIAL:
            if op in OPS["PADDED"]:
                out = self._func_replace_paddding(func, types, args, kwargs)
            elif op in OPS["INTERPOLATE"]:
                out = self._func_interpolate(func, types, args, kwargs)
            elif op in OPS["BATCHED"]:
                out = self._func_batched(func, types, args, kwargs)
            elif op in OPS["CHANNELONLY"]:
                if "dim" in kwargs and kwargs["dim"] == 1:
                    pass
                else:
                    print(f"Operation {op} might behave differently with TensorWrapper when dim != 1!")
                out = super().__torch_function__(func, types, args, kwargs)
            elif op in OPS["WARNING"]:
                warnings.warn(f"Operation {op} might behave differently with TensorWrapper!")
                out = super().__torch_function__(func, types, args, kwargs)
            elif op in OPS["INCOMPATIBLE"]:
                raise AttributeError(f"Operation {op} not supported for TensorWrapper!")
            else:
                raise AttributeError
        else:
            out = super().__torch_function__(func, types, args, kwargs)

        try:
            # if TensorWrapper, initialize it
            out.__init_metadata(self)
        except:
            pass
        return out

    def _func_replace_paddding(self, func, types, args, kwargs):
        """
        Replaces padding of the operation by blockpadding
        If we pad the block-sparse representation of TensorWrapper, the zeros in the padding are "incorrect"
        For that reason, we copy the values from the neighboring block (transferred from previous frame if needed)
        into the padding
        """
        if BLOCKPAD_WITH_ZEROES:  # debug mode
            return super().__torch_function__(func, types, args, kwargs)
        args = list(args)
        padding = kwargs.get("padding", None)
        if padding is None:
            padding = args[4] if len(args) > 4 else 0

        zeros = 0
        if isinstance(padding, (tuple, list)):
            zeros = (0, 0)
            if padding[0] != padding[1]:
                raise NotImplementedError(f"Only support equal paddings, got {padding}")
            padding = padding[0]

        if padding > 0:
            data = args[0]
            data_transfer = self._transfer_from_prev()
            if data_transfer is None:
                # first frame as no transferred data, set to 0-size tensor
                _, C, H, W = data.shape
                data_transfer = torch.empty((0, C, H, W), dtype=data.dtype, device=self.device)
            # store the curent data and transferrd data
            self._features.store_features(data, data_transfer, padding)

            grid_idx = self.get_grid_idx()
            mapping_exec = self.get_mapping_exec()
            with timings.env("tensorwrapper/pad", 10):
                args[0] = pad(data, data_transfer, grid_idx, mapping_exec, padding)  # manually pad

            if "padding" in kwargs:
                kwargs["padding"] = zeros
            else:
                args[4] = zeros

            with timings.env("tensorwrapper/pad_func", 11):
                ret = super().__torch_function__(func, types, args, kwargs)
        else:
            with timings.env("tensorwrapper/pad_func0", 11):
                ret = super().__torch_function__(func, types, args, kwargs)
        return ret

    def _func_interpolate(self, func, types, args, kwargs):
        """
        PyTorch bilinear interpolation is very slow small tensor sizes.
        Curiously, trilinear performs much better (tested in Pytorch 1.9),
        therefore replace bilinear with trilinear with one extra dummy dimension
        """
        if kwargs["mode"] == "bilinear":
            args = list(args)
            data = args[0]
            kwargs["mode"] = "trilinear"
            if kwargs["scale_factor"] is not None:
                sf = kwargs["scale_factor"]
                kwargs["scale_factor"] = (1, sf, sf)
            elif kwargs["size"] is not None:
                size = kwargs["size"]
                kwargs["size"] = (data.shape[1], size[0], size[1])
            args[0] = data.as_subclass(torch.Tensor).unsqueeze(0)
            ret = func(*args, **kwargs)
            ret = ret.squeeze(0).as_subclass(TensorWrapper)
        else:
            ret = super().__torch_function__(func, types, args, kwargs)
        return ret

    def _func_batched(self, func, types, args, kwargs):
        """
        Some functions apply operations per batch element (e.g. group norm). However, Tensorwrapper uses the batch dimension as block dimension
        and this results in wrong outputs. This function adds a batch dimension of 1 so that these operations can be executed correctly.
        Only supports execution with batch-size 1.
        """

        args = list(args)

        # convert data from TensorWrapper to torch.Tensor
        data = args[0].as_subclass(torch.Tensor)

        # remember input shape
        in_shape = data.shape

        # switch batch and channel dimension
        data = data.permute(1, 0, 2, 3)

        # merge batch and spatial dimensions and add new batch dimension of 1
        data = data.reshape(data.shape[0], -1).unsqueeze(0).unsqueeze(3)

        # data is 4d tensor (1, channels, batch x height x width, 1)
        args[0] = data
        types = list(types)
        types[0] = type(data)

        # apply operation
        out = super().__torch_function__(func, types, args, kwargs)

        # convert back to block representation
        out = out.as_subclass(torch.Tensor)
        out = out.squeeze(0).view(in_shape[1], in_shape[0], in_shape[2], in_shape[3]).permute(1, 0, 2, 3).contiguous()
        out = out.as_subclass(TensorWrapper)
        return out
