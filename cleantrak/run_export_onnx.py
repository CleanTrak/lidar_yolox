#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
# Modifications copyright (c) 2025 CleanTrak Inc.

import os
import sys
sys.path.append(f"{os.path.dirname(__file__)}/..")

from cleantrak.yolox_utils import get_num_classes_of_yolox_checkpoint

import argparse
from loguru import logger

import torch
from torch import nn

from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module


def make_parser():
    parser = argparse.ArgumentParser("YOLOX onnx deploy")
    parser.add_argument(
        "--output", type=str, default="yolox.onnx", help="output name of models"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--dynamic", action="store_true", help="whether the input shape should be dynamic or not"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument(
        "--decode_in_inference",
        action="store_true",
        help="decode in inference or not"
    )

    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    if args.exp_file is None:
        args.exp_file = f"{os.path.dirname(__file__)}/config/default.py"
    exp = get_exp(args.exp_file)
    exp.num_classes = get_num_classes_of_yolox_checkpoint(args.ckpt)

    model = exp.get_model()

    # load the model state dict
    logger.info(f"Loading checkpoint {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    model.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = args.decode_in_inference

    dummy_input = torch.randn(args.batch_size, 3, exp.test_size[0], exp.test_size[1])
    model_input = "images"
    model_output = "output"
    opset_version = 21

    logger.info(f"Exporting model into {args.output}")
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        input_names=[model_input],
        output_names=[model_output],
        dynamic_axes={model_input: {0: 'batch'},
                      model_output: {0: 'batch'}} if args.dynamic else None,
        opset_version=opset_version
    )
    logger.info(f"generated onnx model named {args.output}")

    import onnx
    from onnxsim import simplify

    # use onnx-simplifier to reduce reduent model.
    logger.info("running onnx simplifier")
    onnx_model = onnx.load(args.output)
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, args.output)
    logger.info(f"generated simplified onnx model named {args.output}")


if __name__ == "__main__":
    main()
