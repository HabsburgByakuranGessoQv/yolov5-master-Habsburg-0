# import torch.nn as nn
# from models.backbone import DarkNet # 假设存在 DarkNet 类，在另一个文件中定义
# from models.neck import SPP, PAN, ASFF # 假设存在 SPP、PAN 和 ASFF 类，在另一个文件中定义
# from models.head import YoloHead # 假设存在 YoloHead 类，在另一个文件中定义
#
# class YoloV8(nn.Module):
#     def __init__(self):
#         super(YoloV8, self).__init__()
#         self.backbone = DarkNet(...)
#         self.spp = SPP(...)
#         self.neck = PAN(...)  # 假设修改 Neck 网络为 PAN 网络
#         self.head = YoloHead(...)
#
#     def forward(self, x):
#         out_backbone = self.backbone(x)
#         out_spp = self.spp(out_backbone)
#         out_neck = self.neck(out_spp)  # 修改网络时需要在 forward 中传递新的输入
#         out_head = self.head(out_neck)
#         return out_head
#
#
# # 示例代码
# model = YoloV8()
# pretrained_dict = torch.load('path/to/pretrained/model.pth')  # 加载预训练模型
# model_dict = model.state_dict()  # 获取当前模型参数字典
# # 将预训练模型中与当前模型参数名称对应的参数值赋值给当前模型
# new_dict = {k: pretrained_dict[k] for k in pretrained_dict.keys() if k in model_dict.keys()}
# model_dict.update(new_dict)
# model.load_state_dict(model_dict)

import torch
from torchsummary import summary
from models.yolo import DetectionModel  # 替换为您的模型定义路径

weights_path = r'E:\STUDYCONTENT\Pycharm\yolov5-master\runs\train\number2\weights\number.pt'

# 加载预训练权重文件
state_dict = torch.load(weights_path, map_location=torch.device('cpu'))

# 创建一个Yolov5模型
model = DetectionModel(cfg=r'E:\STUDYCONTENT\Pycharm\yolov5-master\models\yolov5s.yaml')  # 替换为您的模型定义

# 加载预训练权重
model.load_state_dict(state_dict['model'])

# 获取DetectionModel对象
detection_model = model.model

# 打印DetectionModel的结构
print(detection_model)