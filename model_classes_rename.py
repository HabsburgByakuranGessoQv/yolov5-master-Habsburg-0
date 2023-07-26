import torch
if __name__ == '__main__':
    model = torch.load(r'E:\STUDYCONTENT\Pycharm\yolov5-master\runs\train\tomatoCBAMC3_2\weights\last.pt')
    print(model)
    model['model'].names[0] = '番茄'
    torch.save(model, "weights/new.pt")