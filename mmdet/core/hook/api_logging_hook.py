# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class CheckInvalidLossHook(Hook):
    """Loggs the results to the enpoint specified in the env variable 

    Args:
        interval (int): Checking interval (every k epochs).
            Default: 1.
    """

    def __init__(self, interval=1):
        self.interval = interval

    def after_train_epoch(self, runner):
        if self.every_n_epochs(runner, self.interval):
            print(runner.outputs)