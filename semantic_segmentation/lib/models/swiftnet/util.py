import warnings

import torch
import torch.nn.functional as F
from blockcopy import blockcopy_noblocks
from blockcopy.utils.profiler import timings
from torch import nn as nn

# batchnorm_momentum = 0.01 / 2

upsample = lambda x, size: F.interpolate(x, size, mode='bilinear')
import logging

BN_MOMENTUM = 0.9

def get_n_params(parameters):
    pp = 0
    for p in parameters:
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, batch_norm=True, bn_momentum=BN_MOMENTUM, bias=False, dilation=1,
                 drop_rate=.0, separable=False):
        super(_BNReluConv, self).__init__()
        if batch_norm:
            self.add_module('norm', nn.BatchNorm2d(num_maps_in, momentum=bn_momentum,track_running_stats=True))
        # self.add_module('relu', nn.ReLU(inplace=batch_norm is True))
        self.add_module('relu', nn.ReLU(inplace=False))
        padding = k // 2
        conv_class = SeparableConv2d if separable else nn.Conv2d
        logging.info(f'Using conv type {k}x{k}: {conv_class}')
        self.add_module('conv', conv_class(num_maps_in, num_maps_out, kernel_size=k, padding=padding, bias=bias,
                                           dilation=dilation))
        if drop_rate > 0:
            logging.info(f'Using dropout with p: {drop_rate}')
            self.add_module('dropout', nn.Dropout2d(drop_rate, inplace=False))

class _Upsample(nn.Module):
    def __init__(self, num_maps_in, skip_maps_in, num_maps_out, use_bn=True, k=3, use_skip=True, only_skip=False,
                 detach_skip=False, fixed_size=None, separable=False, bneck_starts_with_bn=True):
        super(_Upsample, self).__init__()
        print(f'Upsample layer: in = {num_maps_in}, skip = {skip_maps_in}, out = {num_maps_out}')
        self.bottleneck = _BNReluConv(skip_maps_in, num_maps_in, k=1, batch_norm=use_bn and bneck_starts_with_bn)
        self.blend_conv = _BNReluConv(num_maps_in, num_maps_out, k=k, batch_norm=use_bn, separable=separable)
        self.use_skip = use_skip
        self.only_skip = only_skip
        self.detach_skip = detach_skip
        logging.info(f'\tUsing skips: {self.use_skip} (only skips: {self.only_skip})')
        self.upsampling_method = upsample
        self.upfunc = upsample

    def forward(self, x, skip):
        with timings.env('module/skip', 3):
            skip = self.bottleneck(skip)
            
        assert not self.detach_skip
        with timings.env('module/upsample', 3):
            x = self.upfunc(x, (x.shape[2]*2, x.shape[3]*2))
            
        with timings.env('module/blend', 3):
            if self.use_skip:
                x += skip
            x = self.blend_conv(x)
        return x

class SpatialPyramidPooling(nn.Module):
    def __init__(self, num_maps_in, num_levels, bt_size=512, level_size=128, out_size=128,
                 grids=(6, 3, 2, 1), square_grid=False, bn_momentum=0.1, use_bn=True, drop_rate=.0,
                 fixed_size=None, starts_with_bn=True):
        super(SpatialPyramidPooling, self).__init__()
        self.fixed_size = fixed_size
        self.grids = grids
        if self.fixed_size:
            ref = min(self.fixed_size)
            self.grids = list(filter(lambda x: x <= ref, self.grids))
        self.square_grid = square_grid
        self.upsampling_method = upsample
        if self.fixed_size is not None:
            self.upsampling_method = lambda x, size: F.interpolate(x, mode='bilinear', size=fixed_size)
            warnings.warn(f'Fixed upsample size', UserWarning)
        self.spp = nn.Sequential()
        self.spp.add_module('spp_bn', _BNReluConv(num_maps_in, bt_size, k=1, bn_momentum=bn_momentum,
                                                  batch_norm=use_bn and starts_with_bn))
        num_features = bt_size
        final_size = num_features
        for i in range(num_levels):
            final_size += level_size
            self.spp.add_module('spp' + str(i),
                                _BNReluConv(num_features, level_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn,
                                            drop_rate=drop_rate))
        self.spp.add_module('spp_fuse',
                            _BNReluConv(final_size, out_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn))

    @blockcopy_noblocks
    def forward(self, x):
        with timings.env('module/spp_center', 3):
                
            levels = []
            target_size = self.fixed_size if self.fixed_size is not None else x.size()[2:4]

            ar = target_size[1] / target_size[0]

            x = self.spp[0].forward(x)
            levels.append(x)
            num = len(self.spp) - 1

            for i in range(1, num):
                if not self.square_grid:
                    grid_size = (self.grids[i - 1], max(1, round(ar * self.grids[i - 1])))
                    x_pooled = F.adaptive_avg_pool2d(x, grid_size)
                else:
                    x_pooled = F.adaptive_avg_pool2d(x, self.grids[i - 1])
                level = self.spp[i].forward(x_pooled)
                level = self.upsampling_method(level, target_size)
                levels.append(level)

            x = torch.cat(levels, 1)
            x = self.spp[-1].forward(x)
        return x
