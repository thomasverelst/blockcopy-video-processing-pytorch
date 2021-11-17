''' 
for cupy 
'''

from collections import namedtuple
import cupy
import torch
from string import Template
import time
import torch
import ctypes
Stream = namedtuple('Stream', ['ptr'])

# Specify CUDA include path
CUDA_PATH = ('-I/usr/local/cuda/include','-I/usr/local/cuda-11.1/include','-I/usr/local/cuda-11.3/include')

def Dtype(t):
    if t.dtype == torch.float32:
        return 'float'
    elif t.dtype == torch.float16:
        return '__half'
    else:
        raise NotImplementedError(t.dtype)

@cupy.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    options = list(CUDA_PATH[:]).extend(('--restrict','--use_fast_math'))
    kernel_code = cupy.cuda.compile_with_cache(code, options=options)
    return kernel_code.get_function(kernel_name)

def GET_BLOCKS(N, NTHREADS):
    return min((N + NTHREADS - 1) // (NTHREADS), 256*256-1)

def roundup(number, multiple, max_ = None):
    out = multiple * (1 + (number - 1) // multiple)
    if max_ is not None:
        return min(max_, out)
    return out

def cudaok(x, dtype=None, dtype2=None, device=None):
    assert x.is_cuda
    if device is not None:
        assert x.device == device
    assert x.is_contiguous()
    assert (dtype is None or x.dtype == dtype or x.dtype == dtype2), (x.dtype, dtype, dtype2)
    return True

_kernel_header_blocks = '''
#include "cuda_fp16.h"

#define DTYPE ${dtype}
#define BS ${block_size}

#define BATCHSIZE ${batch_size}
#define CHANNELS ${channels}
#define WIDTH ${width}
#define HEIGHT ${height}
#define GRID_W (WIDTH/BS)
#define GRID_H (HEIGHT/BS)

#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < n;                                       \
      i += blockDim.x * gridDim.x)

#define CUDA_CHANNEL_LOOP(c)                       \
  for (int c = blockIdx.y * blockDim.y + threadIdx.y; \
  c < CHANNELS; c += blockDim.y * gridDim.y)

'''
