

训练起来非常慢，建议使用预训练模型和小数据集 ，当然用coco的也行 。  
我这里给大家提供了一个小数据集，夸克网盘链接：https://pan.quark.cn/s/4e4c8e9f5f0b     
下载好数据集后的目录  
![image](https://github.com/user-attachments/assets/420aa63f-83ec-4f52-8cde-9293bf8bf11e)

预训练模型下载链接：[GitHub - zylo117/Yet-Another-EfficientDet-Pytorch: The pytorch re-implement of the official efficientdet with SOTA performance in real time and pretrained weights.](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch?tab=readme-ov-file#pretrained-weights-and-benchmark)（被发现了，就是用的别人训练好的，读书人的...怎么能叫...）  

下载好预训练模型的目录（当然不用全下载，下载自己需要的那个就行）  
![image](https://github.com/user-attachments/assets/2be45368-c145-4115-b2e9-77bb80d7ce35)

如果你要从头训练自己的数据集（非常不推荐）

```
# 设置参数
num_epochs = 500
lr = 1e-5
compound_coef = 0  # 看你需求，只学习的话建议为0
```

使用预训练权重训练自己的数据集（强烈推荐）

```
# 设置参数
num_epochs = 10
lr = 1e-3
compound_coef = 2  # 还是那句话 看需求
load_weights = weights/efficientdet-d2.pth
head_only = True  # 如果不是coco数据集设为False，如果用的coco数据集可以选择冻结
```
用自己数据集改一下data/data.yaml的数据就行 coco的我已提供 
![image](https://github.com/user-attachments/assets/d1d00a99-0e69-4709-b7d0-1a6433af3363)


