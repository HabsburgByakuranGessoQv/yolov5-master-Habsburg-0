# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import contextlib
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import *  # noqa
from models.experimental import *  # noqa
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    # 是 YOLOv5 中的检测头部（Detection Head）。这个类用于处理检测模型的输出，并将其转换成目标检测的预测结果
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes int，表示目标类别的数量（不包括背景类）
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid tuple，包含各个检测层使用的锚框尺寸
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment) bool，是否使用原地操作（inplace ops）

    def forward(self, x):
        """
        模型的前向传播方法，接受输入 x，是来自不同检测层的特征图列表。在推理模式下，该函数会将输出特征图转换为目标检测的预测结果
        Args:
            x: 来自不同检测层的特征图列表

        Returns:
            在推理模式下（self.training = False），forward 方法会返回预测的检测结果，以 (boxes, ) 或 (boxes, features) 的形式
            在训练模式下（self.training = True），forward 方法不返回预测结果，而是返回处理后的特征图 x
        """
        z = []  # inference output z 初始化为空列表，用于存储推理时的输出结果
        for i in range(self.nl):
            # 对于每个检测层，先通过卷积层 self.m[i] 处理对应的特征图 x[i]
            x[i] = self.m[i](x[i])  # conv
            # 将 x[i] 的形状从 (bs,255,20,20) 转换为 (bs,3,20,20,85)，其中 bs 是 batch size，255 是特定的通道数，20 是特定的高度和宽度，85 是特定的预测目标数量和属性数
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            # 使用 view 函数将 x[i] 进行形状变换，变换为 (bs, self.na, self.no, ny, nx)，其中 self.na 是 anchor boxes 的数量，self.no 是预测的目标属性数量，ny 和 nx 是特定的高度和宽度。
            # 使用 permute 函数对维度进行重排列，变换为 (bs, 1, ny, nx, self.no)，其中第二个维度 1 对应于 anchor boxes 的数量 self.na
            # 使用 contiguous 函数使数据在内存中连续存储，以便进行后续计算
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            # 将处理后的特征图 x[i] 转换为预测框的形式，然后将其存储到 z 列表中。如果是 Segment 类型（即含有掩码信息），则对预测框和掩码进行处理；否则只处理预测框
            if not self.training:  # inference 推理模式
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                # 如果是含有掩码信息的 Segment 类型，将特征图 x[i] 分割为 xy、wh、conf 和 mask 四部分，并根据预定义的计算公式进行转换，然后拼接成预测结果 y
                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                # 如果只含有预测框信息（不含掩码），将特征图 x[i] 分割为 xy、wh 和 conf 三部分，并根据预定义的计算公式进行转换，然后拼接成预测结果 y
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                # 将预测结果 y 展平后添加到列表 z 中
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1), ) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        # 一个内部方法，用于生成检测层的网格坐标和锚框偏移。这些信息在推理模式下用于计算目标检测的预测框坐标
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        # 创建网格坐标 grid，它的形状为 (1, self.na, ny, nx, 2)，其中 self.na 是锚框的数量。网格坐标用于计算预测框的中心坐标
        shape = 1, self.na, ny, nx, 2  # grid shape
        # 根据输入参数 nx 和 ny，创建形状为 (ny, nx) 的网格坐标 y 和 x
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        # 创建锚框偏移 anchor_grid，它的形状与网格坐标 grid 相同，但是每个网格点上的值是根据锚框和步长 self.stride[i] 计算得到的。锚框偏移用于计算预测框的宽度和高度
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        # 返回生成的网格坐标 grid 和锚框偏移 anchor_grid
        return grid, anchor_grid


class Segment(Detect):
    # YOLOv5 Segment head for segmentation models
    # 用于分割模型的 YOLOv5 段落头部。这个类定义了一个特定于分割任务的头部结构
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        """
            __init__ 方法是构造函数，用于初始化 Segment 类的对象。它接受一些参数，包括：
                nc：类别的数量，默认为 80（COCO 数据集的类别数）。
                anchors：锚框的尺寸，默认为空元组。
                nm：掩模（分割）的数量，默认为 32。
                npr：原型的数量，默认为 256。
                ch：来自不同检测层的特征图的通道数，默认为空元组。
                inplace：是否使用原地操作，默认为 True
        """
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor 表示每个锚框的输出数量，由于是分割模型，所以是 5（中心坐标和宽高） + nc（类别数量） + nm（掩模数量）
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv 是一个由多个 nn.Conv2d 组成的模块列表，用于生成输出的卷积层
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos 是一个 Proto 类的对象，用于生成掩模的原型
        self.detect = Detect.forward # 用于调用父类 Detect 的前向传播方法

    def forward(self, x):
        # 型的前向传播方法。它接受一个特征图列表 x，来自不同检测层的特征图。在推理模式下，这个方法将输出特征图转换为目标检测的预测结果和掩模的原型。
        p = self.proto(x[0]) # 调用原型的前向传播方法，生成掩模的原型
        x = self.detect(self, x) # 调用父类 Detect 的前向传播方法，生成目标检测的预测结果
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])



class BaseModel(nn.Module):
    # YOLOv5模型的基础类，定义了模型的前向传播、模型层的融合（fuse）、模型信息打印（info）和模型的设备切换（_apply）等功能
    # YOLOv5 base model
    def forward(self, x, profile=False, visualize=False):
        # 前向传播方法，接受输入 x，并返回模型的输出。通过调用 _forward_once 方法进行单尺度推理（inference）或训练
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        # 单次前向传播方法，遍历模型的每一层并依次执行前向运算，同时保存每一层的输出。如果 profile 参数为True，
        # 会记录模型每一层的执行时间和FLOPs，用于性能分析。如果 visualize 参数为True，将对每一层的输出进行可视化
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _profile_one_layer(self, m, x, dt):
        # 性能分析辅助方法，用于分析单个层的执行时间和FLOPs。该方法使用了thop库来计算FLOPs，需要注意在运行之前需要确保thop库已经安装。
        # 该方法在每一层的前向运算中重复执行10次，并记录每次执行的时间，然后计算平均执行时间。同时，它会输出每一层的时间、FLOPs和参数数量
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x, ), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        # 模型融合方法，用于融合模型中的Conv2d()和BatchNorm2d()层，从而加速模型的推理速度。
        # 融合后的模型不包含BatchNorm2d层，而是将其与Conv2d层合并。此方法会更新模型的Conv2d层，并删除BatchNorm2d层
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        # 打印模型的信息，包括每一层的结构、参数数量和输入输出大小等。verbose 参数控制打印详细信息的级别，img_size 参数用于指定输入图像的大小
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        # 设备切换方法，用于将模型的参数和缓冲区（buffer）切换到指定的设备。该方法重写了父类的 _apply 方法，确保模型的所有张量都能正确切换到目标设备
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class DetectionModel(BaseModel):
    # YOLOv5 detection model
    # 继承自 BaseModel 的类，表示 YOLOv5 目标检测模型。这个类定义了 YOLOv5 目标检测模型的结构和前向传播过程
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        # 如果 cfg 是字典，则表示已经提供了模型的配置信息，直接使用。否则，通过读取 cfg 文件来获取模型的配置信息，并保存在 self.yaml 中
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model 定义模型
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels 设置输入通道数，默认为配置文件中的通道数
        # 如果提供了 nc 并且与配置文件中的类别数不一致，会通过日志信息覆盖配置文件中的类别数
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        # 如果提供了 anchors，会通过日志信息覆盖配置文件中的锚框信息
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        # 通过 parse_model 函数根据配置文件创建模型，并保存在 self.model 中。同时，将通道数 ch 传递给模型
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names 设置默认的类别名称列表
        self.inplace = self.yaml.get('inplace', True) # 获取是否使用原地操作的标志，默认为 True

        # Build strides, anchors
        # 如果模型的最后一层是 Detect 或 Segment 类型，会计算其 stride 和调整 anchors 的尺寸
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride # 记录了模型最后一层的 stride，即每个检测层的特征图和输入图像之间的尺寸比例
            self._initialize_biases()  # only run once

        # Init weights, biases 初始化模型的权重和偏置
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        # 模型的前向传播方法。它接受一个输入张量 x，以及一些可选的标志，用于控制推理过程
        # 如果 augment 为 True，则进行增强推理（augmented inference）
        if augment:
            return self._forward_augment(x)  # augmented inference, None 用于执行增强推理，返回增强推理的结果和 None
        # 如果 augment 为 False，则进行单尺度推理（single-scale inference）或训练
        return self._forward_once(x, profile, visualize)  # single-scale inference, train 用于执行单尺度推理或训练，返回推理或训练的结果

    def _forward_augment(self, x):
        # 执行增强推理，使用不同的尺度和翻转组合来进行数据增强，并返回增强推理的结果和 None
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation) 用于对预测结果进行反向缩放，以适应原始图像尺寸
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails 用于对增强推理的结果进行裁剪，以去除冗余的预测框
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency 用于初始化模型的偏置（biases），以便在训练过程中更好地适应不同的数据集和类别
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(datasets.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:5 + m.nc] += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

# 最后，Model = DetectionModel 将 DetectionModel 赋值给 Model，以保持向后兼容性
Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility


class SegmentationModel(DetectionModel):
    # YOLOv5 segmentation model
    # 用于创建分割模型的子类，继承自DetectionModel
    def __init__(self, cfg='yolov5s-seg.yaml', ch=3, nc=None, anchors=None):
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    # YOLOv5 classification model
    # 创建分类模型的子类，继承自BaseModel
    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):  # yaml, model, number of classes, cutoff index
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        # 创建分类模型的方法
        # Create a YOLOv5 classification model from a YOLOv5 detection model
        if isinstance(model, DetectMultiBackend):
            # 将对其进行解包，得到实际的检测模型
            model = model.model  # unwrap DetectMultiBackend
        # 然后根据给定的cutoff参数，截断模型的最后几层，只保留backbone部分
        model.model = model.model[:cutoff]  # backbone
        # 最后，将Classify类（表示分类层）添加到模型的最后一层
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        # 从YAML文件创建分类模型的方法。在该方法中，模型设置为None，因为从YAML文件创建模型的具体逻辑不在此处实现。
        # 实际的YAML文件解析和模型构建可能在_from_detection_model方法中完成
        # Create a YOLOv5 classification model from a *.yaml file
        self.model = None


def parse_model(d, ch):  # model_dict, input_channels(3)    该函数接受一个包含模型配置的字典d 和输入通道数ch
    # Parse a YOLOv5 model.yaml dictionary
    # 对 YoloV5 的 model.yaml 中的字典结构 进行语法分析整理, 并构建模型的网络层
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    # 函数首先从配置字典中提取 anchors（锚框）、nc（类别数）、gd（深度缩放因子）、gw（宽度缩放因子）和激活函数类型。
    anchors, nc, gd, gw, act = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
    if act:
        # 如果配置文件指定了激活函数，会根据配置文件的函数去加载, 比如： Conv.default_act = nn.SiLU()
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5) 输出通道数量 = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    # 接下来，函数遍历模型的 backbone（骨干网络）和 head（头部网络）配置，逐个解析并构建模型的层
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        # 导入模块
        # 对于每个层，函数根据模块类型和参数创建相应的 PyTorch 模块对象，并将其添加到模型的层列表 layers 中
        if m in {
                Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x,
                CBAMC3, Conv_CBAM, SE, ECA, CoordAtt, NAMC3}: # Addition: CBAMC3, Conv_CBAM, CA, ECA, SE, CoordAtt
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x,
                     CBAMC3, NAMC3}: # Addition: CBAMC3 NAMC3
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # TODO: channel, gw, gd
        elif m in {Detect, Segment}:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, 8)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        # NAMAttention
        elif m in [NAMAttention]:  # channels  # CrissCrossAttention bug,
            c1 = ch[f]
            args = [c1]
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        # 在创建模块对象之前，函数根据一些规则对通道数 ch 进行更新
        if i == 0:
            ch = []
        ch.append(c2)
    # 最后，函数返回一个包含所有层的 nn.Sequential 对象和一个按顺序排列的需要保存的层的索引列表。
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    # YOLOv5m summary: 212 layers, 21172173 parameters, 21172173 gradients, 48.9 GFLOPs
    # YOLOv5s summary: 157 layers, 7225885 parameters, 7225885 gradients, 16.4 GFLOPs
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str,
                        default=
                        # 'yolov5s.yaml'
                        # 'yolov5m.yaml'
                        # 'yolov5m_NAMAttention_1.yaml'
                        # 'yolov5m_NAMC3_1.yaml'
                        'yolov5m_CBAMC3-NAMA.yaml'
                        # 'yolov5m_CBAMC3+NAMC3.yaml'
                        # 'yolov5mCBAMC3_1.yaml'
                        ,
                        help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)

    # Options
    # 如果 opt.line_profile 为 True，则执行 model(im, profile=True) 对模型进行逐层速度分析
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    # 如果 opt.profile 为 True，则使用 profile 函数对前向传播和反向传播进行速度分析
    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    # 如果 opt.test 为 True，则测试所有 yolo*.yaml 配置文件，并打印出错误信息（如果有）
    elif opt.test:  # test all models
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    # 如果 opt.test 为 True，则测试所有 yolo*.yaml 配置文件，并打印出错误信息（如果有）
    else:  # report fused model summary
        model.fuse()
