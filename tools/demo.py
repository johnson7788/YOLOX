#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger

import cv2

import torch

from yolox.data.data_augment import preproc
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

IMAGE_EXT = ['.jpg', '.jpeg', '.webp', '.bmp', '.png']


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument('demo', default='image', help='文件的类型，支持 image, video and webcam')
    parser.add_argument("-expn", "--experiment-name", type=str, default=None,help="此次实验的名称")
    parser.add_argument("-n", "--name", type=str, default=None, help="模型的名称，和-f相斥，只用一个")

    parser.add_argument('--path', default='./assets/dog.jpg', help='文件的路径，例如./assets/dog.jpg')
    parser.add_argument('--camid', type=int, default=0, help='如果是摄像头，指定摄像头的id')
    parser.add_argument(
        '--save_result', action='store_true',
        help='是否保存image/video推理结果'
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="指定exp文件，数据加载的一些配置和模型的一些配置",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint的路径，用于推理")
    parser.add_argument("--device", default="cpu", type=str, help="使用的设备，例如 cpu or gpu")
    parser.add_argument("--conf", default=None, type=float, help="置信度")
    parser.add_argument("--nms", default=None, type=float, help="nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="图像的尺寸img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="是否采用混合精度进行评估",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="使用 TensorRT model for testing.",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(self, model, exp, cls_names=COCO_CLASSES, trt_file=None, decoder=None, device="cpu"):
        self.model = model
        self.cls_names = cls_names   #标签的名称的，COCO_CLASSES是80分类的
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        if trt_file is not None:
            from torch2trt import TRTModule
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img):
        img_info = {'id': 0}
        if isinstance(img, str):
            img_info['file_name'] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info['file_name'] = None

        height, width = img.shape[:2]
        img_info['height'] = height
        img_info['width'] = width
        img_info['raw_img'] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info['ratio'] = ratio
        img = torch.from_numpy(img).unsqueeze(0)
        if self.device == "gpu":
            img = img.cuda()

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
                if self.device == "gpu":
                    outputs = outputs.cpu().numpy()
            outputs = postprocess(
                        outputs, self.num_classes, self.confthre, self.nmsthre
                    )
            logger.info('Infer time: {:.4f}s'.format(time.time()-t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info['ratio']
        img = img_info['raw_img']
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


def image_demo(predictor, vis_folder, path, current_time, save_result):
    """
    图像的预测
    :param predictor: 实例化的类，预测器
    :type predictor:
    :param vis_folder: 保存的结果目录 './YOLOX_outputs/yolox_s/vis_res'
    :type vis_folder:
    :param path:  'assets/dog.jpg'
    :type path:
    :param current_time: 当前的时间，开始预测前的时间，用于计时
    :type current_time:
    :param save_result:  True
    :type save_result: bool
    :return:
    :rtype:
    """
    if os.path.isdir(path):
        #如果给的是目录，预测目录下所有图
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        #推理图片
        outputs, img_info = predictor.inference(image_name)
        # 把bbox画到图片上
        result_image = predictor.visual(outputs[0], img_info)
        if save_result:
            #保存推理的结果
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("保存目标检测的结果到图片: {}".format(save_file_name))
            # 保存图片
            cv2.imwrite(save_file_name, result_image)
        # ch = cv2.waitKey(0)
        # if ch == 27 or ch == ord('q') or ch == ord('Q'):
        #     break


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == 'video' else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_folder = os.path.join(vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = os.path.join(save_folder, args.path.split('/')[-1])
    else:
        save_path = os.path.join(save_folder, 'camera.mp4')
    logger.info(f'video save_path is {save_path}')
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(width), int(height))
    )
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0], img_info)
            if args.save_result:
                vid_writer.write(result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord('q') or ch == ord('Q'):
                break
        else:
            break


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name
    # file_name: './YOLOX_outputs/yolox_s' 输出目录
    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)
    # 创建保存的结果目录，vis_res
    if args.save_result:
        vis_folder = os.path.join(file_name, 'vis_res')
        os.makedirs(vis_folder, exist_ok=True)
    #是否使用TensorRT模型
    if args.trt:
        args.device="gpu"
    # 打印参数
    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)
    # 加载初始化模型
    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    #是否使用GPU
    if args.device == "gpu":
        model.cuda()
    #模型是评估模式
    model.eval()
    #不使用TensorRT模型的话
    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("开始加载模型的 checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("加载模型 checkpoint 完成.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert (not args.fuse),\
            "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(trt_file), (
            "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        )
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None
    # 开始预测
    predictor = Predictor(model, exp, COCO_CLASSES, trt_file, decoder, args.device)
    #计算下时间
    current_time = time.localtime()
    if args.demo == 'image':
        # vis_folder: 保存结果的目录， args.path： 图片的目录， args.save_result：bool，是否保存结果
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
    elif args.demo == 'video' or args.demo == 'webcam':
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    #解析命令行参数
    args = make_parser().parse_args()
    #获取实验的配置
    exp = get_exp(args.exp_file, args.name)
    #运行
    main(exp, args)
