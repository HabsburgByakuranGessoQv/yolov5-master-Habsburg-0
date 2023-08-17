import torch
if __name__ == '__main__':
    model_path = r'E:\STUDYCONTENT\Pycharm\yolov5-master\weights\yolov5m6.pt'
    model = torch.load(model_path)
    print(model)
    print(model['model'].names[0])
    # model['model'].names[0] = '番茄'
    # rename = model_path[:-3] + '_renamed.pt'
    # print(rename)
    # torch.save(model, rename)