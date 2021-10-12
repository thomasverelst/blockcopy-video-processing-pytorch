
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.profiler import timings
from torch.autograd import Function

from .cuda import (GET_BLOCKS, Dtype, Stream, _kernel_header_blocks, cudaok,
                   load_kernel, roundup)

CUDA_NUM_THREADS = 512
CUDA_NUM_BLOCKS = 2048
        
def pad(features, transfer, grid_idx, exec_map, pad=1):
    if False: # debug mode with zero pad
        return F.pad(features, (pad,)*4)

    return BlockPadFunction().apply(features, transfer, grid_idx, exec_map, pad)


class BlockPadFunction(Function):
    @staticmethod
    def forward(ctx, data_exec, data_transfer, grid_idx, mapping_exec, pad):
        assert cudaok(data_exec, torch.float16, torch.float32)
        assert cudaok(data_transfer, torch.float16, torch.float32)
        assert cudaok(mapping_exec, torch.int32)
        assert cudaok(grid_idx, torch.int32)

        assert data_exec.shape[1:] == data_transfer.shape[1:], (data_exec.shape, data_transfer.shape)
        assert len(mapping_exec) <= data_exec.shape[0]
        assert grid_idx.numel() - len(mapping_exec) <= data_transfer.shape[0], (grid_idx.numel(), len(mapping_exec), data_transfer.shape)
        assert pad > 0        

        N, _, grid_height, grid_width = grid_idx.shape
        B,C,blocksize,_ = data_exec.shape
        assert blocksize > 0 

        if blocksize <= 2:
            import warnings
            warnings.warn('Block size of 2 or smaller is can be inefficient!')
        
        blocksize_padded = blocksize+2*pad
        H, W = blocksize*grid_height, blocksize*grid_width
        
        out_size = (B, C, blocksize_padded, blocksize_padded)
        out = torch.empty(out_size, device=data_exec.device, dtype=data_exec.dtype)

        npixels_out = len(mapping_exec) * (blocksize_padded*blocksize_padded)
        if npixels_out > 0:
            threads_x = roundup(npixels_out, CUDA_NUM_BLOCKS, CUDA_NUM_THREADS)
            threads_y = max(1, CUDA_NUM_THREADS//threads_x)
            block = (threads_x, threads_y)
            grid_x = min(GET_BLOCKS(npixels_out, threads_x), CUDA_NUM_BLOCKS)
            grid_y = max(1, min(CUDA_NUM_BLOCKS//grid_x+1, GET_BLOCKS(C, threads_y))) 
            grid = (grid_x, grid_y)

            with timings.env('block/pad_kernel',20):
                with torch.cuda.device_of(data_exec):
                    f = load_kernel('repad_kernel', _repad_kernel,
                                    dtype=Dtype(data_exec),
                    block_size=blocksize, batch_size=N, channels=C, height=H, width=W, pad=pad)

                    f(block=block, grid=grid,
                        args=[
                            out.data_ptr(),
                            data_exec.data_ptr(), data_transfer.data_ptr(),  grid_idx.data_ptr(),
                            mapping_exec.data_ptr(), npixels_out,
                        ],stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return out

    @staticmethod
    def backward(ctx, grad_x):
        raise NotImplementedError('Backward not implemented for BlockPad')

_repad_kernel = _kernel_header_blocks+'''
#define PAD ${pad}
#define BS_PAD (BS+2*PAD)

extern "C" __global__ void repad_kernel(
    DTYPE* __restrict__ const out, 
    const DTYPE* __restrict__ const features, 
    const DTYPE* __restrict__ const transfer, 
    const int* const grid_idx, 
    const int* const exec_map, 
    const int npixels){

CUDA_KERNEL_LOOP(i, npixels){
    const int b_pad = i / (BS_PAD*BS_PAD);     // exec block idx
    const int h_pad = (i / BS_PAD) % BS_PAD;   // row
    const int w_pad = i % BS_PAD;              // column
    const int i_b = b_pad*CHANNELS*BS_PAD*BS_PAD + h_pad*BS_PAD + w_pad; // index of channel 0 in blocks

    int b = b_pad;
    int h = h_pad - PAD;
    int w = w_pad - PAD;
    
    const DTYPE* data = features;

    // check if this position is in patch padding 
    const bool left = w_pad<PAD;
    const bool right = w_pad>=BS_PAD-PAD;
    const bool top = h_pad<PAD;
    const bool bottom = h_pad>=BS_PAD-PAD;

    bool zero_pad = false;
    if(left||right||top||bottom){
        // in padding

        // find position of block it is in
        const int g_id = exec_map[b]; // linear patch id
        
        const bool grid_left = g_id % GRID_W == 0;
        const bool grid_right = g_id % GRID_W == GRID_W-1;
        const bool grid_top =  (g_id % (GRID_H * GRID_W)) < GRID_W;
        const bool grid_bottom = (g_id % (GRID_H * GRID_W)) >= (GRID_H * GRID_W) - GRID_W;
        
        zero_pad = (left & grid_left) || (right & grid_right) ||
                     (top & grid_top) || (bottom & grid_bottom);
        
        if(!zero_pad){
            // pad by copying from neighbour
        
            int g_id_in = g_id;
            g_id_in += (right - left);
            g_id_in += GRID_W*(bottom -top);

            int b_in = grid_idx[g_id_in];
            const bool is_exec = b_in >= 0;
            if(!is_exec) b_in += BATCHSIZE*GRID_H*GRID_W;

            if(left){
                w = BS - PAD + w_pad;
            }else if(right){
                w = w_pad - BS_PAD + PAD;
            }
            if(top){
                h = BS - PAD + h_pad;
            }else if(bottom){
                h = h_pad - BS_PAD + PAD;
            }

            data = (is_exec) ? features : transfer;
            b = b_in;
        }
    }

    CUDA_CHANNEL_LOOP(c){ // loop over channels
        DTYPE val = zero_pad ? (DTYPE) 0 : data[b*CHANNELS*BS*BS + c*BS*BS + h*BS + w];
        out[i_b + c*BS_PAD*BS_PAD] = val;
    }

} // close kernel loop
} // close kernel
'''
