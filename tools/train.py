#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn

from yolox.core import Trainer, launch
from yolox.exp import get_exp
from yolox.utils import configure_nccl


def make_parser():
    parser = argparse.ArgumentParser("YOLOX 训练配置parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None,help="此次实验的名字")
    parser.add_argument("-n", "--name", type=str, default=None, help="模型的名字，是exps/default/ 下面的名字中的一个，和-f手动指定相斥，使用-n或-f中的一个即可")

    # 分布式
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="使用哪种分布式协议"
    )
    parser.add_argument(
        "--dist-url", default=None, type=str, help="url used to set up distributed training"
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=1, type=int, help="使用几个GPU设备进行训练"
    )
    parser.add_argument(
        "--local_rank", default=0, type=int, help="分布式训练，使用哪个rank"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="指定配置脚本，一些模型的和数据的配置，参考exps/example/yolox_voc/yolox_voc_s.py",
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="是否继续训练"
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="加载checkpoint文件，继续训练")
    parser.add_argument(
        "-e", "--start_epoch", default=None, type=int, help="继续训练时，从哪个epoch开始计数"
    )
    parser.add_argument(
        "--num_machine", default=1, type=int, help="训练时的node数量"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="多node训练时，node的 rank"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=True,
        action="store_true",
        help="是否采用混合精度训练",
    )
    parser.add_argument(
        "-o",
        "--occumpy",
        dest="occumpy",
        default=False,
        action="store_true",
        help="训练开始前就占用GPU内存，防止显存分片",
    )
    parser.add_argument(
        "opts",
        help="通过命令行来修改的config选项",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    configure_nccl()
    cudnn.benchmark = True
    #初始化和训练， exp是训练的参数，args是命令传过来的参数
    trainer = Trainer(exp, args)
    trainer.train()


if __name__ == "__main__":
    #解析配置
    args = make_parser().parse_args()
    # 获取哪个配置文件，按照名字获取，还是按照文件获取，文件优先
    exp = get_exp(args.exp_file, args.name)
    # 参数合并
    exp.merge(args.opts)
    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    # GPU的使用数量
    assert num_gpu <= torch.cuda.device_count()
    #分布式url
    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main, num_gpu, args.num_machine, backend=args.dist_backend,
        dist_url=dist_url, args=(exp, args)
    )
