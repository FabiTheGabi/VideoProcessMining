#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Optimizer."""

import torch

import slowfast.utils.lr_policy as lr_policy


def construct_optimizer(model, cfg):
    """
    Construct a stochastic gradient descent or ADAM optimizer with momentum.
    Details can be found in:
    Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
    and
    Diederik P.Kingma, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."

    Args:
        model (model): model to perform stochastic gradient descent
        optimization or ADAM optimization.
        cfg (config): configs of hyper-parameters of SGD or ADAM, includes base
        learning rate,  momentum, weight_decay, dampening, and etc.
    """

    if cfg.TRAIN.FINETUNE:
        # Use seperate learning rates for
        head_params = []
        other_layer_params = []
        for name, p in model.named_parameters():
            if is_requires_grad_parameter(cfg, name):
                # Separate parameter group for head with higher
                if "head." in name:
                    head_params.append(p)
                else:
                    other_layer_params.append(p)

        if head_params:
            if other_layer_params:
                optim_params = [
                    {"params": head_params, "lr": cfg.SOLVER.BASE_LR, "weight_decay": cfg.SOLVER.WEIGHT_DECAY,  "name": "head_params"},
                    {"params": other_layer_params, "lr": cfg.SOLVER.BASE_LR/cfg.TRAIN.FINETUNE_BASE_LR_REDUCTION_FACTOR,
                     "weight_decay": cfg.SOLVER.WEIGHT_DECAY, "name": "other_layer_params"},
                ]
            else:
                optim_params = [
                    {"params": head_params, "lr": cfg.SOLVER.BASE_LR, "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
                     "name": "head_params"},
                ]
        else:
            if other_layer_params:
                optim_params = [
                    {"params": other_layer_params, "lr": cfg.SOLVER.BASE_LR/cfg.TRAIN.FINETUNE_BASE_LR_REDUCTION_FACTOR, "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
                     "name": "other_layer_params"},
                ]

    else:
        # Batchnorm parameters.
        bn_params = []
        # Non-batchnorm parameters.
        non_bn_parameters = []
        for name, p in model.named_parameters():
            # If we do not finetune, we add all parameters
            if "bn" in name:
                bn_params.append(p)
            else:
                non_bn_parameters.append(p)
        # Apply different weight decay to Batchnorm and non-batchnorm parameters.
        # In Caffe2 classification codebase the weight decay for batchnorm is 0.0.
        # Having a different weight decay on batchnorm might cause a performance
        # drop.
        optim_params = [
            {"params": bn_params, "weight_decay": cfg.BN.WEIGHT_DECAY},
            {"params": non_bn_parameters, "weight_decay": cfg.SOLVER.WEIGHT_DECAY},
        ]
        # Check all parameters will be passed into optimizer.
        assert len(list(model.parameters())) == len(non_bn_parameters) + len(
            bn_params
        ), "parameter size does not match: {} + {} != {}".format(
            len(non_bn_parameters), len(bn_params), len(list(model.parameters()))
        )

    if cfg.SOLVER.OPTIMIZING_METHOD == "sgd":
        return torch.optim.SGD(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            dampening=cfg.SOLVER.DAMPENING,
            nesterov=cfg.SOLVER.NESTEROV,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adam":
        return torch.optim.Adam(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg.SOLVER.OPTIMIZING_METHOD)
        )


def get_epoch_lr(cur_epoch, cfg):
    """
    Retrieves the lr for the given epoch (as specified by the lr policy).
    Args:
        cfg (config): configs of hyper-parameters of ADAM, includes base
        learning rate, betas, and weight decays.
        cur_epoch (float): the number of epoch of the current training stage.
    """
    return lr_policy.get_lr_at_epoch(cfg, cur_epoch)


def set_lr(optimizer, new_lr, cfg):
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
        cfg: the config file
    """
    for param_group in optimizer.param_groups:
        if cfg.TRAIN.FINETUNE:
            # Set potentially lower lr value to this param_group
            if param_group["name"] == "head_params":
                param_group["lr"] = new_lr
            else:
                param_group["lr"] = new_lr/cfg.TRAIN.FINETUNE_BASE_LR_REDUCTION_FACTOR

        else:
            param_group["lr"] = new_lr

def is_requires_grad_parameter(cfg, parameter_name):
    """
    Determines, if a parameter requires_grad (should be trained) depending on the model architecture
    :param cfg: the config file
    :param parameter_name: the name of the parameter
    :return:
    """
    if cfg.TRAIN.FINETUNE:
        # Contains all the parameter names that require_grad
        requires_grad_parameter_list = []

        if cfg.MODEL.ARCH == "slow":
            # Use only the head
            requires_grad_parameter_list = ["head.projection.weight", "head.projection.bias"]
            if not cfg.TRAIN.FINETUNE_HEAD_ONLY:
                # Use also last ResTage
                for unfreeze_param in cfg.TRAIN.FINETUNE_UNFREEZE_PARAM_LIST:
                    requires_grad_parameter_list.append(unfreeze_param)

        elif cfg.MODEL.ARCH == "slowfast":
            # Use only the head
            requires_grad_parameter_list = ["head.projection.weight", "head.projection.bias"]
            if not cfg.TRAIN.FINETUNE_HEAD_ONLY:
                # Use also last ResTage
                for unfreeze_param in cfg.TRAIN.FINETUNE_UNFREEZE_PARAM_LIST:
                    requires_grad_parameter_list.append(unfreeze_param)  # could also add the "_fuse" for temporal changes

        # If the parameter is in
        for requires_grad_parameter in requires_grad_parameter_list:
            if requires_grad_parameter in parameter_name:
                # We change this parameter in the finetuning process
                return True
        # We do not change this parameter in the finetuning process
        return False
    else:
        return True