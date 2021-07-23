# 训练自定义的数据集.
示例是微调YOLOX-S模型，使用的是VOC数据集

## 0. 开始前，配置YOLOX
参照 [README](../README.md) 安装 YOLOX.

## 1. 创建你的数据集
**Step 1** 首先准备好你自己的带有图像和标签的数据集。为了给图像贴标签，你可以使用一个工具，如 [Labelme](https://github.com/wkentaro/labelme) or [CVAT](https://github.com/openvinotoolkit/cvat).

**Step 2** 然后，你应该编写相应的数据集类，它可以通过"\_\_getitem\_"方法加载图像和标签。我们目前支持COCO格式和VOC格式。

你也可以自己编写数据集。我们以[VOC](./yolox/data/datasets/voc.py#L151)数据集文件为例:
```python
    @Dataset.resize_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

        return img, target, img_info, img_id
```

还有一点值得注意的是，你还应该为Mosiac和MixUp数据增强"[pull_item](.../yolox/data/datasets/voc.py#L129) "和"[load_anno](.../yolox/data/datasets/voc.py#L121) 方法。

**Step 3** 准备评估员。我们目前有 [COCO evaluator](../yolox/evaluators/coco_evaluator.py) and [VOC evaluator](../yolox/evaluators/voc_evaluator.py).
如果你有自己的格式数据或评估指标，你可以编写自己的评估器。

**Step 4** 将您的数据集放在 $YOLOX_DIR/datasets$下，对于 VOC：
```shell
ln -s /path/to/your/VOCdevkit ./datasets/VOCdevkit
```
* 路径 "VOCdevkit "将在下一节描述的exp文件中使用。特别是在 "get_data_loader "和 "get_eval_loader "函数中。

## 2. 创建exp文件以控制所有内容
我们把模型中涉及的所有内容都放在一个单一的Exp文件中，包括模型设置、训练设置和测试设置。

**A complete Exp file is at [yolox_base.py](../yolox/exp/yolox_base.py).** It may be too long to write for every exp, but you can inherit the base Exp file and only overwrite the changed part.

Let's still take the [VOC Exp file](../exps/example/yolox_voc/yolox_voc_s.py) for an example.

We select YOLOX-S model here, so we should change the network depth and width. VOC has only 20 classes, so we should also change the num_classes.

These configs are changed in the init() methd:
```python
class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 20
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
```

Besides, you should also overwrite the dataset and evaluator preprared before to training the model on your own data.

Please see "[get_data_loader](../exps/example/yolox_voc/yolox_voc_s.py#L20)", "[get_eval_loader](../exps/example/yolox_voc/yolox_voc_s.py#L82)", and "[get_evaluator](../exps/example/yolox_voc/yolox_voc_s.py#L113)" for more details.

## 3. Train
Except special cases, we always recommend to use our [COCO pretrained weights](../README.md) for initializing.

Once you get the Exp file and the COCO pretrained weights we provided, you can train your own model by the following command:
```bash
python tools/train.py -f /path/to/your/Exp/file -d 8 -b 64 --fp16 -o -c /path/to/the/pretrained/weights
```

or take the YOLOX-S VOC training for example:
```bash
python tools/train.py -f exps/example/yolox_voc/yolox_voc_s.py -d 8 -b 64 --fp16 -o -c /path/to/yolox_s.pth.tar
```

(Don't worry for the different shape of detection head between the pretrained weights and your own model, we will handle it)

## 4. Tips for Best Training Results

As YOLOX is an anchor-free detector with only several hyper-parameters, most of the time good results can be obtained with no changes to the models or training settings.
We thus always recommend you first train with all default training settings.

If at first you don't get good results, there are steps you could consider to take to improve.

**Model Selection** We provide YOLOX-Nano, YOLOX-Tiny, and YOLOX-S for mobile deployments, while YOLOX-M/L/X for cloud or high performance GPU deployments.

If your deployment meets some trouble of compatibility. we recommand YOLOX-DarkNet53.

**Training Configs** If your training overfits early, then you can reduce max\_epochs or decrease the base\_lr and min\_lr\_ratio in your Exp file:
```python
# --------------  training config --------------------- #
    self.warmup_epochs = 5
    self.max_epoch = 300
    self.warmup_lr = 0
    self.basic_lr_per_img = 0.01 / 64.0
    self.scheduler = "yoloxwarmcos"
    self.no_aug_epochs = 15
    self.min_lr_ratio = 0.05
    self.ema = True

    self.weight_decay = 5e-4
    self.momentum = 0.9
```

**Aug Configs** You may also change the degree of the augmentations.

Generally, for small models, you should weak the aug, while for large models or small size of dataset, you may enchance the aug in your Exp file:
```python
# --------------- transform config ----------------- #
    self.degrees = 10.0
    self.translate = 0.1
    self.scale = (0.1, 2)
    self.mscale = (0.8, 1.6)
    self.shear = 2.0
    self.perspective = 0.0
    self.enable_mixup = True
```

**Design your own detector** You may refer to our [Arxiv](https://arxiv.org/abs/2107.08430) paper for details and suggestions for designing your own detector.
