import torch
if __name__ == '__main__':
    model_path = r'E:\STUDYCONTENT\Pycharm\yolov5-master\runs\train\Q_Tomato_5mCBAMC3_hypDe_noMulti-scale_2_GoodDataset\weights\best.pt'
    model = torch.load(model_path)
    print(model)
    model['model'].names[0] = '番茄'
    rename = model_path[:-3] + '_renamed.pt'
    print(rename)
    torch.save(model, rename)