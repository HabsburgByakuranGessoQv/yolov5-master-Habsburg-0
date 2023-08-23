import torch
if __name__ == '__main__':
    model_path = r'E:\STUDYCONTENT\Pycharm\yolov5-master\runs\train\5mCBAM_w5m6_hypHi_Mulit_1_14+12\weights\last.pt'
    model = torch.load(model_path)
    # print(model)

    # print(model['model'].names[0])
    print(model['epoch'])

    # model['model'].names[0] = '番茄'
    model['epoch'] = 0

    # rename = model_path[:-3] + '_renamed.pt'
    rename = model_path[:-3] + '_lowepoch.pt'

    print(rename)

    torch.save(model, rename)