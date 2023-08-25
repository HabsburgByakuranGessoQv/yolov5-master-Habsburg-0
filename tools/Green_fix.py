# 该文件是一个将图片的rgb 3个通道的均值调整到一个固定值的程序.
import cv2
import numpy as np


def shift_channels(image, r_shift, g_shift, b_shift):
    # 分割图像的三个颜色通道
    b, g, r = cv2.split(image)

    # 平移每个通道的像素值
    b = np.clip(b.astype(np.int32) + b_shift, 0, 255).astype(np.uint8)
    g = np.clip(g.astype(np.int32) + g_shift, 0, 255).astype(np.uint8)
    r = np.clip(r.astype(np.int32) + r_shift, 0, 255).astype(np.uint8)

    # 合并三个通道得到调整后的图像
    shifted_image = cv2.merge([b, g, r])

    return shifted_image


# 读取图像
image_main = cv2.imread(r'D:\StuData\pest\5055_638120089624152649_FTP.jpg')

# 计算图像每个通道的平均值
b_mean_main = np.mean(image_main[:, :, 0])
g_mean_main = np.mean(image_main[:, :, 1])
r_mean_main = np.mean(image_main[:, :, 2])

# 计算每个通道的平移量
b_shift_main = int(128 - b_mean_main)
g_shift_main = int(128 - g_mean_main)
r_shift_main = int(128 - r_mean_main)

print(b_mean_main, g_mean_main, r_mean_main)

# 调用平移通道函数
balanced_image = shift_channels(image_main, r_shift_main, g_shift_main, b_shift_main)


# 显示结果图像
cv2.imshow('Original Image', image_main)
cv2.imshow('Balanced Image', balanced_image)
cv2.imwrite('balanced.jpg', balanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
