# Prepare datasets

如果你有一个数据集目录，你可以使用名为`YOLOX_DATADIR`的os环境变量。在这个目录下，如果需要，YOLOX将按照下面描述的结构寻找数据集。

```
$YOLOX_DATADIR/
  COCO/
```
您可以设置内置数据集的位置, 通过环境变量
```shell
export YOLOX_DATADIR=/path/to/your/datasets
```
如果没有设置`YOLOX_DATADIR`，数据集目录的默认值是相对于你当前工作目录的`./datasets`。

## 预期的数据集结构 [COCO detection](https://cocodataset.org/#download):

```
COCO/
  annotations/
    instances_{train,val}2017.json
  {train,val}2017/
    # 相应的JSON中提到的图像文件
```
您也可以使用2014年版本的数据集。