#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified in the VideoProcessMining project

"""Wrapper to train and test a video classification model."""
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args

from demo_original_net import demo_original
from demo_net import run_demo
from test_net import test
from train_net import train
from visualization import visualize


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)

    if cfg.DEMO.ENABLE:
        original_num_gpus = cfg.NUM_GPUS
        # Set num_gpus to 1 for the demo
        cfg.NUM_GPUS = 1
        launch_job(cfg=cfg, init_method=args.init_method, func=run_demo)
        # Set num gpus back to original
        cfg.NUM_GPUS = original_num_gpus

    # Perform training.
    if cfg.TRAIN.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=train)

    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)

    if cfg.DEMO_ORIGINAL.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=demo_original)

    if cfg.TENSORBOARD.ENABLE and cfg.TENSORBOARD.MODEL_VIS.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=visualize)


if __name__ == "__main__":
    main()
