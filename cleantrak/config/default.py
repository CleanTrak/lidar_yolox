# Copyright (c) 2021 Megvii Inc. All rights reserved.
# Modifications copyright (c) 2025 CleanTrak Inc. All rights reserved.

# from yolox.exp.default.yolox_tiny import Exp as BaseExp
from exps.default.yolox_tiny import Exp as BaseExp

class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        # cleantrak_logic: Uses RGB and treats uses self.train_images as path to images,
        # and self.train_ann, self.val_ann as paths to .json annotations, the same for validation set
        self.cleantrak_logic = True
        # self.test_size = (384, 640)
        self.input_size = self.test_size
        # self.depth = 0.33
        # self.width = 0.50
        self.no_aug_epochs = self.max_epoch // 30 + 5
        self.eval_interval = 5
        self.print_interval = self.eval_interval
        self.hsv_prob = 0.0
        # self.mosaic_prob = 0.0
        # self.enable_mixup = False
        self.test_ann = None
        self.num_classes = None
        self.mosaic_scale = (0.5, 1.5)  # what is this?
        self.random_size = (10, 20)  # what is this?
        self.exp_name = "cleantrak"
