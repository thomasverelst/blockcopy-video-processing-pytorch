import abc
import logging
from abc import abstractmethod
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from blockcopy.policy.information_gain import (InformationGain, InformationGainObjectDetection,
                                               InformationGainSemSeg)
from blockcopy.policy.net import PolicyNet, build_policy_net_from_settings
from ..utils.profiler import timings
from torch.distributions import Bernoulli, Categorical


def build_policy_from_settings(settings: Dict) -> None:
    policy_name = settings['block_policy']
    logging.info(f"> Policy: {policy_name} with execution percentage target {settings['block_target']} and block size {settings['block_size']}")
    
    if policy_name == 'all':
        return PolicyAll(block_size=settings['block_size'], verbose=settings['block_policy_verbose'])
    if policy_name == 'none':
        return PolicyNone(block_size=settings['block_size'], verbose=settings['block_policy_verbose'])
    if policy_name == 'random':
        return PolicyRandom(block_size=settings['block_size'], verbose=settings['block_policy_verbose'])
    if policy_name.startswith('rl_'): # reinforcement learnign policy
        net = build_policy_net_from_settings(settings)
        optimizer = build_policy_optimizer_from_settings(settings, net)
        if policy_name == 'rl_semseg':
            information_gain = InformationGainSemSeg(num_classes=settings['block_num_classes'])
        elif policy_name == 'rl_objectdetection':
            information_gain = InformationGainObjectDetection(num_classes=settings['block_num_classes'])
        else:
            raise AttributeError(f'Policy with name "{policy_name}" not defined!')
        
        return PolicyTrainRL(block_size=settings['block_size'], block_target=settings['block_target'],
                            cost_momentum=settings['block_cost_momentum'], optimizer=optimizer,
                            complexity_weight = settings['block_complexity_weight'],
                            policy_net=net, information_gain=information_gain, verbose=settings['block_policy_verbose'])

    raise NotImplementedError(f"Policy {policy_name} not implemented")

def build_policy_optimizer_from_settings(settings: Dict, net: PolicyNet) -> torch.optim.Optimizer:
    # return torch.optim.Adam(net.parameters(), lr=settings['block_optim_lr'], 
    #                            weight_decay=settings['block_optim_wd'])
    return torch.optim.RMSprop(net.parameters(), lr=settings['block_optim_lr'], 
                               weight_decay=settings['block_optim_wd'], centered=False, 
                               momentum=settings['block_optim_momentum'])


class PolicyStats:
    def __init__(self):
        self.count_images = 0
        self.exec = 0
        self.total = 0
    
    def add_policy_meta(self, policy_meta):
        grid = policy_meta['grid']
        num_exec = int(grid.sum())
        num_total = int(grid.numel())
        policy_meta['num_exec'] = num_exec
        policy_meta['num_total'] = num_total
        policy_meta['perc_exec'] = float(num_exec)/num_total

        self.count_images += grid.size(0)
        self.exec += num_exec
        self.total += num_total

        return policy_meta
    
    def get_exec_percentage(self):
        return float(self.exec)/self.total

    def __repr__(self) -> str:
        return f'Policy stats: average exec percentage {self.get_exec_percentage()}'
        

class Policy(torch.nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, block_size, verbose=False):
        super().__init__()
        self.block_size = block_size
        self.net = None
        self.optimizer = None
        self.verbose = verbose
        self.stats = PolicyStats()
        self.fp16_enabled = False

    
    def is_trainable(self):
        return self.net is not None

    @abstractmethod
    def forward(self, policy_meta):
        raise NotImplementedError

    def optim(self, policy_meta, train=True, **kwargs):
        return policy_meta


class PolicyAll(Policy):
    """
    Execute all blocks
    """
    def forward(self, policy_meta):
        N, C, H, W = policy_meta['inputs'].shape
        assert H % self.block_size == 0, f"input height ({H}) not a multiple of block size {self.block_size}!"
        assert W % self.block_size == 0, f"input width  ({W}) not a multiple of block size {self.block_size}!"

        G = (H//self.block_size, W//self.block_size)
        grid = torch.ones((N, 1, G[0], G[1]), device=policy_meta['inputs'].device, dtype=torch.bool)
        policy_meta['grid'] = grid
        policy_meta = self.stats.add_policy_meta(policy_meta)
        return policy_meta
    
class PolicyNone(Policy):
    """
    Execute no blocks
    """
    def forward(self, policy_meta):
        N, C, H, W = policy_meta['inputs'].shape
        assert H % self.block_size == 0, f"input height ({H}) not a multiple of block size {self.block_size}!"
        assert W % self.block_size == 0, f"input width  ({W}) not a multiple of block size {self.block_size}!"

        G = (H//self.block_size, W//self.block_size)
        if policy_meta.get('outputs_prev', None) is None:
            grid = torch.ones( (N, 1, G[0], G[1]), device=policy_meta['inputs'].device).type(torch.bool)
        else:
            grid = torch.zeros( (N, 1, G[0], G[1]), device=policy_meta['inputs'].device).type(torch.bool)
        policy_meta['grid'] = grid
        policy_meta = self.stats.add_policy_meta(policy_meta)
        return policy_meta

class PolicyRandom(Policy):
    """
    Execute random blocks
    """        
    def forward(self, policy_meta):
        frame = policy_meta['inputs']
        N,C,H,W = frame.shape
        G = (H//self.block_size, W//self.block_size)
        assert H % self.block_size == 0, f"input height ({H}) not a multiple of block size {self.block_size}!"
        assert W % self.block_size == 0, f"input width  ({W}) not a multiple of block size {self.block_size}!"

        if policy_meta.get('outputs_prev', None) is None:
            grid = torch.ones( (N, 1, G[0], G[1]), device=policy_meta['inputs'].device).type(torch.bool)
        else:
            grid = (torch.randn( (N, 1, G[0], G[1]), device=policy_meta['inputs'].device) > 0).type(torch.bool)            
        policy_meta['grid'] = grid
        policy_meta = self.stats.add_policy_meta(policy_meta)
        return policy_meta
    
class PolicyTrainRL(Policy, metaclass=abc.ABCMeta):
    def __init__(self, block_size: int, block_target: float, cost_momentum: float,  optimizer: torch.optim.Optimizer, 
    complexity_weight : float,
    policy_net: PolicyNet, information_gain: InformationGain, num_init_images: int = 100, verbose: bool = False):
        super().__init__(block_size, verbose)
        assert block_target <= 1
        assert block_target >= 0
        self.block_target = block_target
        self.information_gain = information_gain
        
        self.momentum = cost_momentum
        self.running_cost = None
        self.net = policy_net

        self.num_init_images = num_init_images

        self.complexity_weight_gamma = complexity_weight
        self.optimizer = optimizer
    
    def forward(self, policy_meta):
        N,C,H,W = policy_meta['inputs'].shape
        assert H % self.block_size == 0, f"input height ({H}) not a multiple of block size {self.block_size}!"
        assert W % self.block_size == 0, f"input width  ({W}) not a multiple of block size {self.block_size}!"

        if policy_meta['outputs'] is None:
            # if no temporal history, execute all
            G = (H//self.block_size, W//self.block_size)
            grid = torch.ones( (N, 1, G[0], G[1]), device=policy_meta['inputs'].device, dtype=torch.bool)
            policy_meta['grid'] = grid            
        else:
            # execute policy net
            with torch.enable_grad():   
                self.net.train()
                grid_logits = self.net(policy_meta)
                assert torch.all(~torch.isnan(grid_logits)), "Policy net returned NaN's, maybe optimization problem?"
                
                m = Bernoulli(logits=grid_logits) # create distribution
                grid = m.sample() # sample
                
                # if grid.sum() == 0:
                    # if no blocks executed, execute a single one
                    # grid[0,0,0,0] = 1
                
                grid_probs = m.probs
                grid_log_probs = m.log_prob(grid)
                
                assert grid.dim() == 4
                assert grid_probs.dim() == 4
                assert grid_probs.shape == grid.shape
                
                policy_meta['grid_log_probs'] = grid_log_probs
                policy_meta['grid_probs'] = grid_probs
                policy_meta['grid'] = grid.bool()
        policy_meta = self.stats.add_policy_meta(policy_meta)
        return policy_meta

    def _get_information_gain(self, policy_meta: Dict) -> torch.Tensor:
        with timings.env('policy/information_gain', 3):
            ig = self.information_gain(policy_meta)
            assert ig.dim() == 4
            return ig
    
    def _get_reward_complexity(self, policy_meta: Dict) -> float:
        # anneal target cost over init images to initalize policy
        target =  1-(1-self.block_target)*min(1, self.stats.count_images/self.num_init_images) 
        reward_sparsity = -float(self.running_cost - target)
        reward_sparsity = reward_sparsity*abs(reward_sparsity)
        return reward_sparsity

    def optim(self, policy_meta: Dict, train=True) -> Dict: 
        policy_meta['output_repr'] = self.information_gain.get_output_repr(policy_meta)

        grid = policy_meta['grid']
        assert grid.dim() == 4
        block_use = policy_meta['perc_exec']
        if self.running_cost is None:
            self.running_cost = block_use
        self.running_cost = self.running_cost*self.momentum + (1-self.momentum)*block_use
        
        if policy_meta['outputs_prev'] is not None and train:
            with torch.enable_grad():       
                
                ig = self._get_information_gain(policy_meta)
                policy_meta['information_gain'] = ig
                reward_complexity_weighted = self._get_reward_complexity(policy_meta)*self.complexity_weight_gamma 
                reward = ig + reward_complexity_weighted
                assert reward.dim() == 4      
                assert not torch.any(torch.isnan(reward))
                log_probs = policy_meta['grid_log_probs']
                reward = F.adaptive_max_pool2d(reward, output_size=log_probs.shape[2:])
                reward[~grid] = -reward[~grid]
                loss = -log_probs * reward.detach()
                loss_policy = loss.mean()
                assert not torch.isnan(loss_policy)
            
                # optim
                with timings.env('policy/optimizer_backward', 3):
                    loss_policy.backward()
                with timings.env('policy/optimizer_step', 3):
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                if self.verbose:
                    s = ''
                    s += f'BLOCKS/running_cost: {self.running_cost} \n'
                    s += f'BLOCKS/block_use: {block_use} \n'
                    s += f'BLOCKS/information_gain_max: {ig.max()} \n'
                    s += f'BLOCKS/information_gain_min: {ig.min()} \n'
                    s += f'BLOCKS/reward_complexity_weighted: {reward_complexity_weighted} \n'
                    s += f'BLOCKS/avg_prob_exec: {policy_meta["grid_probs"][grid].mean()} \n'
                    s += f'BLOCKS/avg_prob_skip: {policy_meta["grid_probs"][~grid].mean()} \n'
                    print(s)
                    print(self.stats)
        return policy_meta
