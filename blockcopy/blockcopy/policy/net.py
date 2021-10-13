from blockcopy.policy import backbones
from typing import Dict
import torch.nn as nn 
import torch
import torch.nn.functional as F

def build_policy_net_from_settings(settings: Dict):
    return PolicyNet(settings=settings)

class PolicyNet(nn.Module):
    def __init__(self, settings) -> None:
        super().__init__()
        
        self.block_size = settings['block_size']
        self.kernel_size = 3
        self.scale_factor = 0.25*128/self.block_size # TODO
        self.use_frame_state = True
        self.use_prev_output = True
        self.output_num_classes = settings['block_num_classes']
        self.use_prev_grid = True
        
        
        in_channels = 3 # input image RGB
        if self.use_frame_state:
            in_channels += 3
        if self.use_prev_output:
            in_channels += self.output_num_classes
        if self.use_prev_grid:
            in_channels += 1
        
        self.backbone = backbones.__dict__[settings['block_net']](pretrained=False, in_channels=in_channels, width_factor=1)
        
        mid_channels = self.backbone.OUT_CHANNELS//2
        
        strides = [2,2,2]
        head = []
        head.append(self._make_layer(self.backbone.OUT_CHANNELS, mid_channels, kernel_size=self.kernel_size, stride=strides[0]))
        head.append(self._make_layer(mid_channels, mid_channels, kernel_size=self.kernel_size, stride=strides[1]))
        head.append(self._make_layer(mid_channels, 1, kernel_size=self.kernel_size, stride=strides[2], bnrelu=False))
        self.head = nn.Sequential(*head)        
        
    def _make_layer(self, in_channels, out_channels, kernel_size=3, stride=1, bnrelu=True):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2, dilation=1, stride=stride, bias=False))
        if bnrelu:
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=False))
        return nn.Sequential(*layers)
        
    def forward(self, policy_meta: Dict):
        frame = policy_meta['inputs']
        N,C,H,W = frame.shape

        # gather features
        input_features = []
        
        # frame
        assert frame.dim() == 4
        assert frame.size(1) == 3
        
        input_features.append(F.interpolate(frame, scale_factor=self.scale_factor, mode='nearest').float())
        
        if self.use_frame_state:
            # frame state (last executed frame per block)
            frame_state = policy_meta['frame_state']
            input_features.append(F.interpolate(frame_state, size=input_features[0].shape[2:], mode='nearest').float())
            
        if self.use_prev_output:
            assert policy_meta.get('output_repr', None) is not None
            outputs = policy_meta['output_repr']
            assert outputs.dim() == 4
            outputs = F.interpolate(outputs, size=input_features[0].shape[2:], mode='nearest')
            input_features.append(outputs.type(input_features[0].dtype)-0.5)
            
        if self.use_prev_grid:
            assert policy_meta.get('grid', None) is not None
            grid_prev = policy_meta['grid'].type(input_features[0].dtype)
            assert grid_prev.dim() == 4
            grid_prev = F.interpolate(grid_prev, size=input_features[0].shape[2:], mode='nearest')
            input_features.append(grid_prev-0.5)
            
        x = torch.cat(input_features, dim=1).detach()
        x = self.backbone(x)
        logits = self.head(x)

        assert logits.shape == (N, 1, H//self.block_size, W//self.block_size), \
            f'logits shape: {logits.shape}, frame shape: {(N, C, H, W)}, block size: {self.block_size}'
        return logits
