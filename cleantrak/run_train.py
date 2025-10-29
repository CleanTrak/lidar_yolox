#!/usr/bin/env python
# Copyright (c) Megvii, Inc. and its affiliates.
# Copyright (c) 2025 CleanTrak Inc.

import sys
import os

from pycocotools.coco import COCO

from folders import ensure_folder

sys.path.append(f"{os.path.dirname(__file__)}/..")



import yolox.exp
import argparse
from yolox.exp import get_exp, check_exp_value
from yolox.utils import configure_module, get_num_devices
from tools.train import launch, main


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    # parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    # parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment description file",
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "--cache",
        type=str,
        nargs="?",
        const="ram",
        help="Caching imgs to ram/disk for fast training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "-l",
        "--logger",
        type=str,
        help="Logger to be used for metrics. \
                Implemented loggers include `tensorboard`, `mlflow` and `wandb`.",
        default="tensorboard"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    # Added by CleanTrak
    parser.add_argument(
        "--max_epoch", required=True, type=int, help="Training epoch number at which training will be stopped"
    )
    parser.add_argument(
        "--seed", default=0, type=int, help="Seed for deterministic training"
    )
    parser.add_argument("--train_images", type=str, required=True,
                        help="Path to train images (COCO dataset format)")
    parser.add_argument("--train_labels", type=str, required=True,
                        help="Path to train annotations json file (COCO dataset format)")
    parser.add_argument("--val_images", type=str, default=None,
                        help="Path to validation images (COCO dataset format)")
    parser.add_argument("--val_labels", type=str, default=None,
                        help="Path to validation annotations json file (COCO dataset format)")
    parser.add_argument("--output_dir", "--output", type=str, required=True,
                        help="Path to output folder")
    return parser


if __name__ == "__main__":
    configure_module()
    args = make_parser().parse_args()
    if args.exp_file is None:
        args.exp_file = f"{os.path.dirname(__file__)}/config/default.py"
    exp = get_exp(args.exp_file)
    if args.val_images is None:
        args.val_images = args.train_images
    if args.val_labels is None:
        args.val_labels = args.train_labels

    print(f"Train images: {args.train_images}")
    print(f"Train labels: {args.train_labels}")
    print(f"Val images:   {args.val_images}")
    print(f"Val labels:   {args.val_labels}")

    train_coco = COCO(args.train_labels)
    n_classes = len(train_coco.cats)
    exp.num_classes = n_classes

    exp.data_dir = None
    exp.test_ann = None
    exp.train_images = args.train_images
    exp.train_ann = args.train_labels
    exp.val_images = args.val_images
    exp.val_ann = args.val_labels

    exp.max_epoch = args.max_epoch

    exp.seed = args.seed
    print(f"seed:   {exp.seed}")
    # exp.merge(args.opts)
    check_exp_value(exp)

    exp.output_dir = args.output_dir
    ensure_folder(args.output_dir)
    print(f"Output dir: {args.output_dir}")

    args.experiment_name = ""

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    if args.cache is not None:
        exp.dataset = exp.get_dataset(cache=True, cache_type=args.cache)

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args),
    )