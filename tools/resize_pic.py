import os
import cv2

def resize_image(image, target_width, target_height):
    # 获取原始图像的宽度和高度
    original_height, original_width = image.shape[:2]

    # 判断是否为大像素图片
    is_large_image = original_width > target_width and original_height > target_height

    # 根据判断结果，决定是否缩放
    if is_large_image:
        # 缩放大像素图片到指定大小
        print("{0}是大像素图片, 已修改为({1}, {2}, 3)".format(image.shape, target_width, target_height), end='\t')
        resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    else:
        # 对于小像素图片，保持原始大小
        print("{0}不是大像素图片".format(image.shape), end='\t')
        resized_image = image

    return resized_image


def save_fix(target_width, target_height, img_folder, save_path):
    # 判断保存路径是否存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    img_ilst = os.listdir(img_folder)
    img_num = len(img_ilst)

    for img_i in range(img_num):
        if img_ilst[img_i] == 'labels':
            continue

        img_path = os.path.join(img_folder, img_ilst[img_i])
        image = cv2.imread(img_path)

        # 调用图像缩放函数
        resized_image = resize_image(image, target_width, target_height)

        img_id = r'\number_{0}.jpg'.format(img_i)
        save_name = img_id
        save_name = save_path + save_name
        # print(save_name)
        cv2.imwrite(save_name, resized_image)
        print('save done! {0}'.format(save_name))


if __name__ == '__main__':
    # 设置目标宽度和高度
    target_width_main = 640
    target_height_main = 480

    raw_path = r'D:\StuData\numbers\images'
    new_path = r'D:\StuData\numbers\resize'

    save_fix(target_width_main, target_height_main, raw_path, new_path)