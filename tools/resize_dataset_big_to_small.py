# 该文件是用于修改图像文件像素大小, 并且将新的labels文件也一同保存到新目录下的文件.
import os
import cv2


def resize_image(image, target_width, target_height):
    # 获取原始图像的宽度和高度
    original_height, original_width = image.shape[:2]

    scale_x = 1
    scale_y = 1

    # 判断是否为大像素图片
    is_large_image = original_width > target_width and original_height > target_width

    # 根据判断结果，决定是否缩放
    if is_large_image:
        # 缩放大像素图片到指定大小
        print("{0}是大像素图片, 已修改为({1}, {2}, 3)".format(image.shape, target_width, target_height), end='\n')
        resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

    else:
        # 对于小像素图片，保持原始大小
        print("{0}不是大像素图片".format(image.shape), end='\n')
        resized_image = image

    return resized_image, scale_x, scale_y


def adjust_yolo_label(label_path, save_label, scale_x, scale_y):
    # 无需缩放, yolo是按比例来进行标签标记
    with open(label_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    adjusted_lines = []

    for line in lines:
        # 解析YOLO标签的数据
        class_id, center_x, center_y, width, height = map(float, line.strip().split())

        # 根据缩放比例调整相对坐标为绝对坐标
        center_x *= scale_x
        center_y *= scale_y
        width *= scale_x
        height *= scale_y

        # 将调整后的数据转换回相对坐标形式
        adjusted_line = f"{int(class_id)} {center_x} {center_y} {width} {height}\n"
        adjusted_lines.append(adjusted_line)

    # 将调整后的标签写回文件
    with open(save_label, 'w', encoding='utf-8') as file:
        print('label write done!', end='\t')
        file.writelines(adjusted_lines)


def save_fix(target_width, target_height, folder, save_path):
    # 判断保存路径是否存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_label = os.path.join(save_path, 'labels')
    save_img = os.path.join(save_path, 'images')

    if not os.path.exists(save_label):
        os.makedirs(save_label)

    if not os.path.exists(save_img):
        os.makedirs(save_img)

    label_folder = os.path.join(folder, 'labels')
    img_folder = os.path.join(folder, 'images')

    img_ilst = os.listdir(img_folder)
    img_num = len(img_ilst)

    for img_i in range(img_num):
        if img_ilst[img_i] == 'labels':
            continue

        img_path = os.path.join(img_folder, img_ilst[img_i])
        label_path = os.path.join(label_folder, img_ilst[img_i].split('.')[0] + '.txt')

        # print(label_path)

        image = cv2.imread(img_path)

        # 调用图像缩放函数
        resized_image, scale_x, scale_y = resize_image(image, target_width, target_height)

        img_id = r'\{0}_tomato.jpg'.format(img_i)
        label_id = r'\{0}_tomato.txt'.format(img_i)
        save_name = img_id
        save_name = save_img + save_name
        # print(save_name)
        save_label_path = save_label + label_id
        cv2.imwrite(save_name, resized_image)

        # 调整YOLO标签
        # if os.path.exists(label_path):
        adjust_yolo_label(label_path, save_label_path, scale_x, scale_y)
        print(save_label_path)

        print('save done! {0}'.format(save_name))
        print('*' * 100)


if __name__ == '__main__':
    # 设置目标宽度和高度
    target_width_main = 640
    target_height_main = 640

    raw_path = r'D:\StuData\tomato\paddle\dark_big'
    new_path = r'D:\StuData\tomato\paddle\dark_resize'

    save_fix(target_width_main, target_height_main, raw_path, new_path)
