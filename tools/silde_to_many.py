import os

from PIL import Image

def sliding_window(image_path, img_name, window_size=(640, 640), stride=(640, 640), save_path='output'):
    # 打开原始大图片
    img = Image.open(image_path)
    img_width, img_height = img.size

    # 定义滑动窗口的大小和步长
    window_width, window_height = window_size
    stride_x, stride_y = stride

    # 创建保存目录
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 遍历滑动窗口
    for y in range(0, img_height - window_height + 1, stride_y):
        for x in range(0, img_width - window_width + 1, stride_x):
            # 切割图片
            window = img.crop((x, y, x + window_width, y + window_height))

            # 保存切割后的小图片
            window.save(os.path.join(save_path, f"{img_name.split('.')[0]}_window_{x}_{y}.png"))

if __name__ == "__main__":
    image_path = r"D:\StuData\tomato\background_new"  # 替换为你的大图片路径
    img_list = os.listdir(image_path)
    img_num = len(img_list)
    for i in range(img_num):
        if img_list[i] == 'slide':
            continue
        img_i = os.path.join(image_path, img_list[i])
        print(img_i)
        sliding_window(img_i, window_size=(640, 640), stride=(120, 120), save_path=r'D:\StuData\tomato\background_new\slide', img_name=img_list[i])
