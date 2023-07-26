import os

def background_txt_crate(data_path):
    # 生成背景数据集的txt空标签文件
    # 获取文件目录 前提: 在该目录下创建好了images和labels文件夹
    data_list = os.listdir(data_path)

    # 如果没有labels文件夹 就创建一个
    if len(data_list) == 1:
        os.mkdir(os.path.join(data_path, 'labels'))
        data_list = os.listdir(data_path)

    images_path = os.path.join(data_path, data_list[0]) if data_list[0] == 'images' else os.path.join(data_path, data_list[1])
    labels_path = os.path.join(data_path, data_list[1]) if data_list[1] == 'labels' else os.path.join(data_path, data_list[0])
    # print(images_path, labels_path)

    total_img = os.listdir(images_path)
    num = len(total_img)
    # print(total_img)
    list_index = range(num)

    for i in list_index:
        renamed = 'background_leaf' + str(i) + '.'
        label_name =  renamed + 'txt'

        img_name = renamed + total_img[i].split('.')[-1]
        img_old_path = os.path.join(images_path, total_img[i])
        img_new_path = os.path.join(images_path, img_name)
        os.rename(img_old_path, img_new_path)

        # print(label_name)
        file_txt_i = open(os.path.join(labels_path, label_name), 'w')
        file_txt_i.close()
    print('done! sum:{0}'.format(num))

    return 0


if __name__ == '__main__':
    main_path = r'D:\StuData\tomato\kaggle\sum\leaf'
    background_txt_crate(main_path)