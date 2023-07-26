import matplotlib
print(matplotlib.matplotlib_fname())

import torch
if __name__ == '__main__':
    model_path = r'E:\STUDYCONTENT\Pycharm\yolov5-master\runs\train\exp\weights\best.pt'
    model = torch.load(model_path)
    print(model['model'].names[0])
    # print(model)