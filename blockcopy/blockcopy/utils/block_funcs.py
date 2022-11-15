import torch

from .cuda import (cudaok, GET_BLOCKS, roundup, Dtype, load_kernel, Stream, _kernel_header_blocks)
from torch.autograd import Function
from .profiler import timings

CUDA_NUM_THREADS = 512
CUDA_NUM_BLOCKS = 1024

class SplitFunction(Function):
    @staticmethod
    def forward(ctx, blocks, image, mapping_exec, grid_idx):
        """
        copies image content into blocks tensor according to given mapping and grid_idx
        """
        assert cudaok(blocks, dtype=blocks.dtype, device=blocks.device)
        assert cudaok(image, dtype=blocks.dtype, device=blocks.device)
        assert cudaok(mapping_exec, dtype=torch.int32, device=blocks.device)
        assert cudaok(grid_idx, dtype=torch.int32, device=blocks.device)
        
        assert blocks.dim() == 4
        assert image.dim() == 4
        assert blocks.shape[2] == blocks.shape[3]
        _, C, blocksize, _ = blocks.shape
        N,C_img,H,W = image.shape
        assert C == C_img
        
        _, _, grid_height, grid_width = grid_idx.shape
        assert grid_height*blocksize == H, (grid_height, blocksize, H)
        assert grid_width*blocksize == W, (grid_width, blocksize, W)

        B = len(mapping_exec) # number of executed blocks 
        if B > 0:
            npixels = B*blocksize*blocksize # number of pixels to be processed
            threads_x = roundup(npixels, 32, CUDA_NUM_THREADS)
            threads_y = max(1, min(CUDA_NUM_THREADS//threads_x, C))
            
            block = (threads_x, threads_y)
            grid_x = min(GET_BLOCKS(npixels, threads_x), CUDA_NUM_BLOCKS)
            grid_y = max(1, min(CUDA_NUM_BLOCKS//grid_x+1, GET_BLOCKS(C, threads_y))) 
            grid = (grid_x, grid_y)
            with torch.cuda.device_of(blocks):
                f = load_kernel('split_kernel', _split_kernel, dtype=Dtype(blocks),
                block_size=blocksize, batch_size=N, channels=C, height=H, width=W)
                f(block=block, grid=grid,
                    args=[
                        blocks.data_ptr(), image.data_ptr(), mapping_exec.data_ptr(), npixels
                    ],stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return blocks

    @staticmethod
    def backward(ctx, grad_x):
        raise NotImplementedError()



_split_kernel = _kernel_header_blocks+'''
extern "C"
__global__ void split_kernel(
    DTYPE* __restrict__ const blocks, 
    const DTYPE* __restrict__ const image, 
    const int* const mapping_exec, const int npixels
){

CUDA_KERNEL_LOOP(i, npixels){
    const int b = i / (BS*BS);     // exec block idx
    const int h = (i / BS) % BS;   // row
    const int w = i % BS;              // column
    const int i_b = b*CHANNELS*BS*BS + h*BS + w; // index of channel 0 in blocks

    const int i_g = mapping_exec[b];
    const int gn = i_g / (GRID_H*GRID_W);
    const int gh = (i_g / GRID_W) % GRID_H;
    const int gw = i_g % GRID_W;
    const int i_image =  gn*CHANNELS*WIDTH*HEIGHT + (gh*BS + h)*WIDTH + (gw*BS + w);

    CUDA_CHANNEL_LOOP(c){ // loop over channels
        DTYPE val = image[i_image + c*WIDTH*HEIGHT];
        blocks[i_b + c*BS*BS] = val;
    }
} // close kernel loop
} // close kernel
'''

class CombineFunction(Function):
    @staticmethod
    def forward(ctx, blocks, out, grid_idx, mapping_exec):
        assert cudaok(blocks, dtype=blocks.dtype)
        assert cudaok(out,    dtype=blocks.dtype)
        assert cudaok(grid_idx, dtype=torch.int32)
        assert cudaok(mapping_exec, dtype=torch.int32)
        
        
        N,C,H,W = out.shape
        _,_,blocksize,BS2 = blocks.shape
        B = len(mapping_exec)
        _, _, grid_height, grid_width = grid_idx.shape
        
        assert blocksize >= 1
        assert grid_idx.size(1) == 1
        assert grid_idx.size(0) == N
        assert blocksize == BS2
        assert grid_height*blocksize == H
        assert grid_width*blocksize == W

        npixels = B*blocksize*blocksize
        if npixels > 0:
            threads_x = roundup(npixels, 32, CUDA_NUM_THREADS)
            threads_y = min(max(1, CUDA_NUM_THREADS//threads_x), C) 
            block = (threads_x, threads_y)
            grid_x = min(GET_BLOCKS(npixels, threads_x), CUDA_NUM_BLOCKS)
            grid_y = max(1, min(CUDA_NUM_BLOCKS//grid_x+1, GET_BLOCKS(C, threads_y))) 
            grid = (grid_x, grid_y)

            with timings.env('block/combine_kernel', 20):
                with torch.cuda.device_of(blocks):
                    f = load_kernel('combine_kernel', _combine_kernel, dtype=Dtype(blocks),
                    block_size=blocksize, batch_size=N, channels=C, height=H, width=W, npixels=npixels)
                    f(block=block, grid=grid,
                        args=[
                            blocks.data_ptr(), out.data_ptr(), mapping_exec.data_ptr(), npixels
                        ],stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        return out

    @staticmethod
    def backward(ctx, grad_x):
        raise NotImplementedError()

_combine_kernel = _kernel_header_blocks+'''
#define NPIXELS ${npixels} 

extern "C"
__global__ void combine_kernel(
    const DTYPE* __restrict__ const blocks, 
    DTYPE* __restrict__ const out, 
    const int* const mapping_exec,
    const int npixels
    ){

CUDA_KERNEL_LOOP(i, npixels){
    const int b = i / (BS*BS);     // exec block idx
    const int h = (i / BS) % BS;   // row
    const int w = i % BS;          // column
    const int i_b = b*CHANNELS*BS*BS + h*BS + w; // index of channel 0 in blocks

    const int i_g = mapping_exec[b];
    const int gn = i_g / (GRID_H*GRID_W);
    const int gh = (i_g / GRID_W) % GRID_H;
    const int gw = i_g % GRID_W;
    const int i_image =  gn*CHANNELS*WIDTH*HEIGHT + (gh*BS + h)*WIDTH + (gw*BS + w);

    CUDA_CHANNEL_LOOP(c){ // loop over channels
        out[i_image + c*WIDTH*HEIGHT] = blocks[i_b + c*BS*BS];
    }
} // close kernel loop
} // close kernel
'''


class TransferFunction(Function):
    @staticmethod
    def forward(ctx, data_transfer, prev_computed, prev_transfer, grid_idx_prev, transfer_map_prev, padding):
        assert cudaok(data_transfer, torch.float16, torch.float32, device=data_transfer.device)
        assert cudaok(prev_computed, torch.float16, torch.float32, device=data_transfer.device)
        assert cudaok(prev_transfer, torch.float16, torch.float32, device=data_transfer.device)
        assert cudaok(transfer_map_prev, torch.int32, device=data_transfer.device)
        
        assert data_transfer.shape[1:] == prev_computed.shape[1:]
        assert data_transfer.shape[1:] == prev_transfer.shape[1:]
        
        N, _, grid_height, grid_width = grid_idx_prev.shape    
        B, C, blocksize, _ = data_transfer.shape
        H, W = blocksize*grid_height, blocksize*grid_width
        npixels = len(transfer_map_prev)*blocksize*blocksize
        if npixels > 0:        
            threads_x = roundup(npixels, 32, CUDA_NUM_THREADS)
            threads_y = max(1, min(CUDA_NUM_THREADS//threads_x, C))
            block = (threads_x, threads_y)
            grid_x = min(GET_BLOCKS(npixels, threads_x), CUDA_NUM_BLOCKS)
            grid_y = max(1, min(CUDA_NUM_BLOCKS//grid_x+1, GET_BLOCKS(C, threads_y))) 
            grid = (grid_x, grid_y)

            with torch.cuda.device_of(data_transfer):
                f = load_kernel('transfer_kernel', _transfer_kernel, dtype=Dtype(data_transfer),
                block_size=blocksize, batch_size=N, channels=C, height=H, width=W, padding=padding)
                with timings.env('block/transfer_kernel', 20):
                    f(block=block, grid=grid,
                        args=[
                            data_transfer.data_ptr(), prev_computed.data_ptr(), prev_transfer.data_ptr(), 
                            transfer_map_prev.data_ptr(), npixels
                        ],stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return data_transfer

    @staticmethod
    def backward(ctx, grad_data_tansfer):
        raise NotImplementedError()



_transfer_kernel = _kernel_header_blocks+'''
#define USE_ZEROS false // debug option to just copy zeros
#define PADDING ${padding}

extern "C"
__global__ void transfer_kernel(
    DTYPE* __restrict__ const out, //transfer data output
    const DTYPE* __restrict__ const prev_data, // previous executed data tensor
    const DTYPE* __restrict__ const prev_transfer,  // previous transfered data tensor
    const int* const transfer_map,  // transfer map 
    const int npixels){

CUDA_KERNEL_LOOP(i, npixels){
    const int b = i / (BS*BS);     // exec block idx
    const int h = (i / BS) % BS;   // row
    const int w = i % BS;          // column

    if(PADDING >= 0){
        // if amount padding is known
        if(w >= PADDING && w <= BS-PADDING-1 && h >= PADDING && h <= BS-PADDING-1 ){
            // if not in padding, skip, as this value will never be needed
            continue;
        }
    }
    const int i_b = b*CHANNELS*BS*BS + h*BS + w; // index of channel 0 in blocks
    int b_prev = transfer_map[b];

    const bool is_exec = b_prev >= 0;
    if(!is_exec) b_prev += BATCHSIZE*GRID_H*GRID_W;
    const DTYPE* const data = (is_exec) ? prev_data : prev_transfer;

    CUDA_CHANNEL_LOOP(c){ // loop over channels
        out[i_b + c*BS*BS] = (USE_ZEROS) ? (DTYPE) 0 : data[b_prev*CHANNELS*BS*BS + c*BS*BS + h*BS + w];
    }
} // close kernel loop
} // close kernel
'''


