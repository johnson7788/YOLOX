# 自定义数据集进行训练
## 根据voc的数据集文件进行修改
Small 模型
exp文件： exps/example/yolox_pdfall_s.py
x-large 模型
## small模型训练
python tools/train.py  -f exps/example/yolox_pdfall_s.py -d 1 --fp16 -b 64 -o -c model/yolox_s.pth


python tools/train.py  --depth_width large -f exps/example/yolox_pdfall_s.py -d 1 --fp16 -b 64 -o -c model/yolox_s.pth
