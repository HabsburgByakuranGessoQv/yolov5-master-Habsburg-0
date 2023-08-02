# coding:utf-8
# 该文件用于划分数据集并保存在一个数据集txt文件夹, 可以用各种参数进行写入.
# 注意: 其中mode是你使用方式, 可以在ide中直接文件运行, 也可以在cmd中运行, cmd运行模式仅支持单文件夹写入, file模式则默认在多文件夹写入(可以将list中数量改成1个文件夹目录变成单文件夹写入).
import os
import random
import argparse

def main(opt):
    img_path, save_path, train_val_percent, train_percent, write_mode = \
        opt.img_path, opt.txt_path, opt.train_val_percent, opt.train_percent, opt.aORw

    # img_path = opt.img_path
    # save_path = opt.txt_path
    total_img = os.listdir(img_path)
    root_path = os.path.abspath(img_path)
    print('数据集路径: {0}\n保存路径: {1}'.format(img_path, save_path))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    num = len(total_img)
    # print(total_img)
    list_index = range(num)
    print('-'*50, '数据集总数量: ', '{:<4d}'.format(num), '-'*50)

    tv = int(num * train_val_percent)
    tr = int(tv * train_percent)
    print('-' * 50, '训练集总数量: ', '{:<4d}'.format(tr), '-' * 50)
    print('-' * 50, '验证集总数量: ', '{:<4d}'.format(tv-tr), '-' * 50)
    print('-' * 50, '测试集总数量: ', '{:<4d}'.format(num-tv), '-' * 50)
    train_val = random.sample(list_index, tv)
    train = random.sample(train_val, tr)

    if write_mode == 'w':
        print('纯写入模式...\t')
        file_train_val = open(save_path + '/train_val.txt', 'w')
        file_test = open(save_path + '/test.txt', 'w')
        file_train = open(save_path + '/train.txt', 'w')
        file_val = open(save_path + '/val.txt', 'w')

    else:
        print('追加写入模式...\t')
        file_train_val = open(save_path + '/train_val.txt', 'a')
        file_test = open(save_path + '/test.txt', 'a')
        file_train = open(save_path + '/train.txt', 'a')
        file_val = open(save_path + '/val.txt', 'a')

    for i in list_index:
        name = os.path.join(root_path, total_img[i]) + '\n'
        if i in train_val:
            file_train_val.write(name)
            if i in train:
                file_train.write(name)
            else:
                file_val.write(name)
        else:
            file_test.write(name)

    file_train_val.close()
    file_train.close()
    file_val.close()
    file_test.close()
    print('done!')



def mass_folder_crate(datasets_list):
    # 批量化处理
    if not datasets_list:
        raise Exception(' $datasets_list$ is empty! Can parse to txt!')
    for i in datasets_list:
        # print(datasets_list[0])
        mode_write = 'w' if i == datasets_list[0] else 'a'
        i = os.path.join(i, 'images')
        print(i)
        opt = parse_opt()
        opt.img_path = i
        opt.aORw = mode_write
        # print(opt)
        main(opt)

def parse_opt(known=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_path', default=r'D:\StuData\tomato\kaggle\sum\leaf\images', type=str, help='input img path')
    parser.add_argument('--txt_path', default=r'D:\StuData\dataset_txt_folder\tomato', type=str, help='output txt path')
    parser.add_argument('--train_val_percent', default='1.0', type=float, help='train+val:total -- percent') # 训练集和验证集所占比例。 这里没有划分测试集
    parser.add_argument('--train_percent', default='0.8', type=float, help='train:val -- percent') # 训练集所占比例，可自己进行调整
    parser.add_argument('--aORw', default='a', type=str, help='write a mode or wmode') # 追加写入a还是单纯默认写入w
    parser.add_argument('--mode', default='file', type=str, help='cmd or file mode to start, if cmd: only argument in cmd will run') # 用什么方式写入, 如果是cmd则只执行cmd命令行的参数

    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == '__main__':
    opt_main = parse_opt()
    if opt_main.mode == 'file':
        # 该写入方式下目录仅需要到 /images的上一级!
        datasets_list_main = [r'D:\StuData\tomato\dataset_factory\background\apples',
                         r'D:\StuData\tomato\dataset_factory\background\bitter_gourd',
                         r'D:\StuData\tomato\dataset_factory\background\leaf',
                         r'D:\StuData\tomato\dataset_factory\background\orange',
                         r'D:\StuData\tomato\dataset_factory\background\coco128',
                         r'D:\StuData\tomato\dataset_factory\230116cropped',
                         r'D:\StuData\tomato\dataset_factory\Public_dataset_1',
                         r'D:\StuData\tomato\dataset_factory\kaggle1',
                         ]
        # datasets_list_main = []
        mass_folder_crate(datasets_list_main)
        print('-*-'*70, '\n', '该方式为默认文件内设置的多目录写入, 如您是终端运行, 且仅需要单目录写入, 请运行时加入--mode cmd参数!', '\n',
              'This mode is for writing to multiple directories in the default file, if you are running on a terminal and only need to write to a single directory, please add the --mode cmd parameter when running!',
              '\n', '-*-'*70,)
    else:
        main(opt_main)