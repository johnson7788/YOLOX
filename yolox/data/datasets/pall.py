#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Code are based on
# https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py
# Copyright (c) Francisco Massa.
# Copyright (c) Ellis Brown, Max deGroot.
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import os.path
import pickle
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from yolox.evaluators.voc_eval import voc_eval

from .datasets_wrapper import Dataset
from .pall_classes import PALL_CLASSES


class AnnotationTransform(object):

    """
    将一个图像标注转化为一个Bbox Coords和标签索引的张量，用类名和索引的字典查询进行初始化。

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None):
        # 类别到id的索引，class_to_ind: {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}
        self.class_to_ind = class_to_ind or dict(zip(PALL_CLASSES, range(len(PALL_CLASSES))))

    def __call__(self, label_path, height, width):
        """
        Arguments:
            label_path (annotation) : 标签的txt文件
        Returns:
            返回 一个列表，包含了bboxe [bbox coords, class name]
        """
        res = np.empty((0, 5))
        with open(label_path, 'r') as f:
            for line in f:
                if not line:
                    continue
                line_split = line.split(' ')
                assert len(line_split) == 5, "YOLO的标签格式必须是5列，分别是label_id, x, y, xmax, ymax"
                # 获取左上角和右下角的坐标
                label_id, x_center, y_center, width_percent, height_percent = line_split
                x_center, y_center, width_percent, height_percent = float(x_center), float(y_center), float(
                    width_percent), float(height_percent)
                x0 = (x_center - width_percent / 2) * width
                y0 = (y_center - height_percent / 2) * height
                x1 = (x_center + width_percent / 2) * width
                y1 = (y_center + height_percent / 2) * height
                bndbox = [x0,y0,x1,y1,label_id]  #存储bbox，对应着name目标的bbox, "xmin", "ymin", "xmax", "ymax"
                # 返回的结果格式 bbox和label的id的一个列表 [xmin, ymin, xmax, ymax, label_ind]
                res = np.vstack((res, bndbox))  # [xmin, ymin, xmax, ymax, label_ind]
        #返回格式是，嵌套的列表的numpy格式, shape[3,5], 表示这张图片有3个目标，5表示bbox+labelid组成的列表
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class PDFDetection(Dataset):

    """
    VOC数据集加载
    input is image, target is annotation
    Args:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(
        self,
        data_dir,
        image_sets,  #eg: labeled_all_train
        img_size=(416, 416),
        preproc=None,
        target_transform=AnnotationTransform(),   #生成目标检测的类别的名称到id的映射
        dataset_name="VOC0712",
    ):
        super().__init__(img_size)
        # self.root: '/home/wac/johnson/project'
        self.root = data_dir
        self.image_set = image_sets
        self.img_size = img_size
        self.preproc = preproc
        self.target_transform = target_transform
        self.name = dataset_name
        # 标签的路径
        self._annopath = os.path.join("%s","labels", "%s.txt")
        # 图片的路径
        self._imgpath = os.path.join("%s","images","%s.jpg")
        self._classes = PALL_CLASSES
        self.ids = []
        # 读取文件 datasets/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt中的ids，对应着图片的id
        for dir in image_sets:
            # 数据的目录dir
            self._year = dir
            # 图像目录下
            train_path = os.path.join(self.root, dir)
            image_path = os.path.join(train_path, "images")
            images = os.listdir(image_path)
            for image in images:
                image_id = image.rstrip('.jpg')
                self.ids.append((train_path, image_id))
    def __len__(self):
        return len(self.ids)

    def load_anno(self, index, height, width):
        """
        加载图像的标签, 图像的宽度和高度
        :param index:
        :type index:
        :return:
        :rtype:
        """
        # img_id： 图像的名字
        img_id = self.ids[index]
        # 解析xml， eg：'/home/wac/johnson/johnson/YOLOX/datasets/VOCdevkit/VOC2007/Annotations/000552.xml'
        label_path = self._annopath % img_id
        assert os.path.exists(label_path), "标签不存在，请检查"
        target = self.target_transform(label_path, height, width)

        return target

    def pull_item(self, index):
        """在混合索引处返回原始图像和目标

        Note: not using self.__getitem__(), as any transformations passed in could mess up this functionality.

        Argument:
            index (int): 图像的索引
        Return:
            img, target
        """
        # self.ids，图像的名字， index是图像的位置索引
        img_id = self.ids[index]
        # 读取图像，img，shape： (331, 500, 3)
        img_path = self._imgpath % img_id
        assert os.path.exists(img_path), "图像不存在，请检查"
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # 图片的高度，宽度
        height, width, _ = img.shape
        # 加载标签
        target = self.load_anno(index, height, width)
        # 图像的信息
        img_info = (width, height)
        # 返回图像的numpy格式，标签：多个（bbox+labelid）组成的列表，img_info是图像的宽度和高度，图像的索引
        return img, target, img_info, index

    @Dataset.resize_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

        return img, target, img_info, img_id

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        self._write_voc_results_file(all_boxes)
        IouTh = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
        mAPs = []
        for iou in IouTh:
            mAP = self._do_python_eval(output_dir, iou)
            mAPs.append(mAP)

        print("--------------------------------------------------------------")
        print("map_5095:", np.mean(mAPs))
        print("map_50:", mAPs[0])
        print("--------------------------------------------------------------")
        return np.mean(mAPs), mAPs[0]

    def _get_voc_results_file_template(self):
        filename = "comp4_det_test" + "_{:s}.txt"
        filedir = os.path.join(self.root, "results", "VOC" + self._year, "Main")
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(VOC_CLASSES):
            cls_ind = cls_ind
            if cls == "__background__":
                continue
            print("Writing {} VOC results file".format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, "wt") as f:
                for im_ind, index in enumerate(self.ids):
                    index = index[1]
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        f.write(
                            "{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(
                                index,
                                dets[k, -1],
                                dets[k, 0] + 1,
                                dets[k, 1] + 1,
                                dets[k, 2] + 1,
                                dets[k, 3] + 1,
                            )
                        )

    def _do_python_eval(self, output_dir="output", iou=0.5):
        rootpath = os.path.join(self.root, "VOC" + self._year)
        name = self.image_set[0][1]
        annopath = os.path.join(rootpath, "Annotations", "{:s}.xml")
        imagesetfile = os.path.join(rootpath, "ImageSets", "Main", name + ".txt")
        cachedir = os.path.join(
            self.root, "annotations_cache", "VOC" + self._year, name
        )
        if not os.path.exists(cachedir):
            os.makedirs(cachedir)
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print("Eval IoU : {:.2f}".format(iou))
        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(VOC_CLASSES):

            if cls == "__background__":
                continue

            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename,
                annopath,
                imagesetfile,
                cls,
                cachedir,
                ovthresh=iou,
                use_07_metric=use_07_metric,
            )
            aps += [ap]
            if iou == 0.5:
                print("AP for {} = {:.4f}".format(cls, ap))
            if output_dir is not None:
                with open(os.path.join(output_dir, cls + "_pr.pkl"), "wb") as f:
                    pickle.dump({"rec": rec, "prec": prec, "ap": ap}, f)
        if iou == 0.5:
            print("Mean AP = {:.4f}".format(np.mean(aps)))
            print("~~~~~~~~")
            print("Results:")
            for ap in aps:
                print("{:.3f}".format(ap))
            print("{:.3f}".format(np.mean(aps)))
            print("~~~~~~~~")
            print("")
            print("--------------------------------------------------------------")
            print("Results computed with the **unofficial** Python eval code.")
            print("Results should be very close to the official MATLAB eval code.")
            print("Recompute with `./tools/reval.py --matlab ...` for your paper.")
            print("-- Thanks, The Management")
            print("--------------------------------------------------------------")

        return np.mean(aps)
