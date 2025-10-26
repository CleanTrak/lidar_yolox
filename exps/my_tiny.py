# Copyright (c) 2021 Megvii Inc. All rights reserved.
# Modifications copyright (c) 2025 CleanTrak Inc. All rights reserved.

from yolox.exp.default.yolox_tiny import Exp as BaseExp

class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.eval_interval = 1
        self.num_classes = 2
        self.data_num_workers = 1
