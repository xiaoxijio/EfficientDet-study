

训练起来非常慢，建议使用预训练模型和小数据集  
我这里给大家提供了一个小数据集，夸克网盘链接：晚点上传  
预训练模型下载链接：[GitHub - zylo117/Yet-Another-EfficientDet-Pytorch: The pytorch re-implement of the official efficientdet with SOTA performance in real time and pretrained weights.](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch?tab=readme-ov-file#pretrained-weights-and-benchmark)（被发现了，就是用的别人训练好的，读书人的...怎么能叫...）

如果你要从头训练自己的数据集（非常不推荐）

```
# 设置参数
num_epochs = 500
lr = 1e-5
compound_coef看你需求，只学习的话建议为0
```

使用预训练权重训练自己的数据集（强烈推荐）

```
# 设置参数
num_epochs = 10
lr = 1e-3
compound_coef = 2  # 还是那句话 看需求
load_weights = weights/efficientdet-d2.pth
```
