import itertools
import torch
import torch.nn as nn
import numpy as np


class BBoxTransform(nn.Module):
    def forward(self, anchors, regression):
        """
        将模型输出的边界框回归值 (dy, dx, dh, dw) 转换为实际图像坐标上的边界框 (xmin, ymin, xmax, ymax)
        decode_box_outputs adapted from https://github.com/google/automl/blob/master/efficientdet/anchors.py

        Args:
            anchors: [batchsize, boxes, (y1, x1, y2, x2)]
            regression: [batchsize, boxes, (dy, dx, dh, dw)]

        Returns:

        """
        dy_centers = (anchors[..., 0] + anchors[..., 2]) / 2
        dx_centers = (anchors[..., 1] + anchors[..., 3]) / 2
        dh = anchors[..., 2] - anchors[..., 0]
        dw = anchors[..., 3] - anchors[..., 1]

        w = regression[..., 3].exp() * dw
        h = regression[..., 2].exp() * dh

        y_centers = regression[..., 0] * dh + dy_centers
        x_centers = regression[..., 1] * dw + dx_centers

        ymin = y_centers - h / 2.
        xmin = x_centers - w / 2.
        ymax = y_centers + h / 2.
        xmax = x_centers + w / 2.

        return torch.stack([xmin, ymin, xmax, ymax], dim=2)


class ClipBoxes(nn.Module):

    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        """

        :param boxes: [batch_size, num_boxes, (xmin, ymin, xmax, ymax)]
        :param img: [batch_size, num_channels, height, width]
        :return:
        """
        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)  # 限制xmin最小值0以上
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)  # 限制ymin最小值0以上

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width - 1)  # 限制xmax最大值width - 1以内
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height - 1)  # 限制ymax最大值height - 1以内

        return boxes


class Anchors(nn.Module):
    """
    adapted and modified from https://github.com/google/automl/blob/master/efficientdet/anchors.py by Zylo117
    """

    def __init__(self, anchor_scale=4., pyramid_levels=None, **kwargs):
        super().__init__()
        self.anchor_scale = anchor_scale

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels

        self.strides = kwargs.get('strides', [2 ** x for x in self.pyramid_levels])
        self.scales = np.array(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        self.ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])

        self.last_anchors = {}
        self.last_shape = None

    def forward(self, image, dtype=torch.float32):
        """
        生成多尺度的锚框
        将原始图像中的锚框位置映射到经过特征提取的多层特征图上的位置
        根据原始图像的尺寸、特征图的步长（stride）、尺度（scales）、宽高比（ratios），生成多层次的锚框坐标
        以便能够在不同的特征图尺度上进行物体检测
        :param image: (batch_size, channels, height, width)
        :param dtype: 锚框的精度
        :return:
        """
        image_shape = image.shape[2:]

        if image_shape == self.last_shape and image.device in self.last_anchors:
            return self.last_anchors[image.device]  # 返回缓存的锚框

        if self.last_shape is None or self.last_shape != image_shape:
            self.last_shape = image_shape

        if dtype == torch.float16:
            dtype = np.float16
        else:
            dtype = np.float32

        boxes_all = []
        for stride in self.strides:  # 遍历特征图的采样步长
            boxes_level = []
            for scale, ratio in itertools.product(self.scales, self.ratios):  # 遍历锚框的大小比例和宽高比
                if image_shape[1] % stride != 0:
                    raise ValueError('input size must be divided by the stride.')
                base_anchor_size = self.anchor_scale * stride * scale  # 锚框大小
                anchor_size_x_2 = base_anchor_size * ratio[0] / 2.0
                anchor_size_y_2 = base_anchor_size * ratio[1] / 2.0

                x = np.arange(stride / 2, image_shape[1], stride)  # 锚框中心点
                y = np.arange(stride / 2, image_shape[0], stride)
                xv, yv = np.meshgrid(x, y)
                xv = xv.reshape(-1)
                yv = yv.reshape(-1)

                # y1,x1,y2,x2
                boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                                   yv + anchor_size_y_2, xv + anchor_size_x_2))
                boxes = np.swapaxes(boxes, 0, 1)
                boxes_level.append(np.expand_dims(boxes, axis=1))
            # concat anchors on the same level to the reshape NxAx4
            boxes_level = np.concatenate(boxes_level, axis=1)  # 同一特征图层的所有锚框拼接在一起
            boxes_all.append(boxes_level.reshape([-1, 4]))

        anchor_boxes = np.vstack(boxes_all)

        anchor_boxes = torch.from_numpy(anchor_boxes.astype(dtype)).to(image.device)
        anchor_boxes = anchor_boxes.unsqueeze(0)

        # save it for later use to reduce overhead
        self.last_anchors[image.device] = anchor_boxes
        return anchor_boxes
