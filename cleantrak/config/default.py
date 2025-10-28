# Copyright (c) 2021 Megvii Inc. All rights reserved.
# Modifications copyright (c) 2025 CleanTrak Inc. All rights reserved.

# from yolox.exp.default.yolox_tiny import Exp as BaseExp
from exps.default.yolox_tiny import Exp as BaseExp

class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        # cleantrak_logic: Uses RGB and treats self.data_dir as path to images, and self.train_ann, self.val_ann as paths to .json annotations
        self.cleantrak_logic = True
        self.test_ann = None
        self.eval_interval = 10
        self.num_classes = None
        self.hsv_prob = 0.0
        self.enable_mixup = False
        # self.data_num_workers = 1
