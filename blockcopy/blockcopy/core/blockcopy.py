import torch
import torch.nn as nn
from ..utils.profiler import timings
import blockcopy


class BlockCopyModel(nn.Module):
    """
    Wrapper around a PyTorch model to enable blockcopy

    inputs:
        base_model: PyTorch model for main task
        settings: dict with settings for blockcopy
    """

    def __init__(self, base_model: nn.Module, settings: dict):
        super().__init__()
        self.is_blockcopy_manager = (
            True  # variable to indicate this Module is managing the BlockCopy state over multiple frames
        )
        self.base_model = base_model  # the main task model
        self.policy = blockcopy.build_policy_from_settings(settings)  # the policy network

        self.block_temporal_features = None  # the features to be saved
        self.reset_temporal()  # reset features
        self.train_interval = settings[
            "block_train_interval"
        ]  # time steps between training the policy (not updating every step, to save computation)

    def load_state_dict(self, state_dict: "OrderedDict[str, Tensor]", strict: bool = True):
        """for backwards compatibility of base_model checkpoints"""
        return self.base_model.load_state_dict(state_dict, strict=strict)

    def reset_temporal(self):
        """
        Resets the state of BlockCopy (e.g. at start of new clip)
        """
        self.clip_length = 0
        if self.block_temporal_features:
            self.block_temporal_features.clear()
        self.block_temporal_features = None
        self.policy_meta = {"inputs": None, "outputs": None, "outputs_prev": None}
        torch.cuda.empty_cache()  # remove old cached values from cuda memory

    def forward(self, inputs, **kwargs):
        return self._forward_blockcopy(inputs, **kwargs)

    def _forward_blockcopy(self, inputs, **kwargs):
        self.clip_length += 1

        # run policy
        self.policy_meta["inputs"] = inputs
        with timings.env("blockcopy/policy_forward", 3):
            # policy adds execution grid is in self.policy_meta['grid']
            self.policy_meta = self.policy(self.policy_meta)

        with timings.env("blockcopy/model", 3):
            # convert inputs into tensorwrapper object
            x = blockcopy.to_tensorwrapper(inputs)

            # run model with block-sparse execution
            if self.policy_meta["num_exec"] == 0:
                # if no blocks to be executed, just copy outputs
                self.policy_meta = self.policy_meta.copy()
                out = self.policy_meta["outputs"]
            else:
                # set meta from previous run to integrate temporal aspects
                self.block_temporal_features = x.process_temporal_features(self.block_temporal_features)

                # convert to blocks with given grid
                inputs = x.to_blocks(self.policy_meta["grid"])

                # get frame state (latest executed frame per block)
                self.policy_meta["frame_state"] = inputs.combine_().to_tensor()

                # run model
                out = self.base_model(inputs, **kwargs)
                # combine blocks into regular tensor
                out = out.combine().to_tensor()

            # keep previous outputs for policy
            self.policy_meta["outputs_prev"] = self.policy_meta["outputs"]
            self.policy_meta["outputs"] = out

        with timings.env("blockcopy/policy_optim", 3):
            if self.policy is not None:
                train_policy = self.clip_length % self.train_interval == 0
                self.policy_meta = self.policy.optim(self.policy_meta, train=train_policy)
        return out


def blockcopy_noblocks(func):
    """
    Decorator to run a torch.nn.Module without blocks.
    Has a large performance cost due to required combine and split.

    example:

    class MyIncompatibleModule(nn.Module):
        @blockcopy_noblocks
        def forward(self, x):
            ...

    """

    def noblocks(self, x: blockcopy.TensorWrapper, *args) -> blockcopy.TensorWrapper:
        """
        Wrapper to run a torch.nn.Module without blocks.
        x: TensorWrapper object
        """
        is_blocks = isinstance(x, blockcopy.TensorWrapper)
        if is_blocks:
            blocks = x
            x = x.combine_().to_tensor()

        x = func(self, x)

        if is_blocks:
            x = blockcopy.to_tensorwrapper(x).to_blocks_like(blocks)
        return x

    return noblocks
