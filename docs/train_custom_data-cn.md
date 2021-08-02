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

*完整的 Exp文件在 [yolox_base.py](../yolox/exp/yolox_base.py).** 为每一个Exp写可能太长了，但你可以继承基本的Exp文件，只覆盖改变的部分。

示例 [VOC Exp file](../exps/example/yolox_voc/yolox_voc_s.py) 

我们在这里选择YOLOX-S模型，所以我们应该改变网络深度和宽度。VOC只有20个类，所以我们也应该改变num_classes。

init（）方法中更改了这些配置：
```python
class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 20
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
```

此外，在对你自己的数据进行训练之前，你还应该重写数据集和评估器。


参考文件 "[get_data_loader](../exps/example/yolox_voc/yolox_voc_s.py#L20)", "[get_eval_loader](../exps/example/yolox_voc/yolox_voc_s.py#L82)", and "[get_evaluator](../exps/example/yolox_voc/yolox_voc_s.py#L113)" for more details.

## 3. 训练
除特殊情况外，我们总是建议使用我们的 [COCO pretrained weights](../README.md) 用于初始化.

一旦你得到Exp文件和我们提供的COCO预训练权重，你就可以通过以下命令训练自己的模型。
```bash
python tools/train.py -f /path/to/your/Exp/file -d 8 -b 64 --fp16 -o -c /path/to/the/pretrained/weights
```
或者yolox-s VOC进行训练：
```bash
python tools/train.py -f exps/example/yolox_voc/yolox_voc_s.py -d 8 -b 64 --fp16 -o -c /path/to/yolox_s.pth.tar
```

(不要担心预训练的权重和你自己的模型之间的检测头的形状不同，我们会处理的。)

## 4.最佳训练结果提示

由于YOLOX是一个只有几个超参数的无锚检测器，大多数情况下，不改变模型或训练设置就可以获得良好的结果。
因此，我们总是建议你先用所有默认的训练设置进行训练。

如果一开始你没有得到好的结果，你可以考虑采取一些措施来改善。

**模型选择**我们提供用于移动部署的YOLOX-Nano、YOLOX-Tiny和YOLOX-S，而YOLOX-M/L/X则用于云或高性能GPU部署。

如果您的部署符合兼容性的一些麻烦。我们推荐yolox-darknet53。

**训练配置** 如果你的训练提前过拟合，那么你可以在Exp文件中减少max/epochs或者减少base/lr和min_lr/ratio。
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

**数据增强配置** 您还可能会更改增强程度。

一般来说，对于小模型，你应该弱化AUG，而对于大模型或小规模的数据集，你可以在Exp文件中加强AUG。
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

**设计你自己的检测器**你可以参考我们的[Arxiv](https://arxiv.org/abs/2107.08430)论文，了解设计你自己检测器的细节和建议。
