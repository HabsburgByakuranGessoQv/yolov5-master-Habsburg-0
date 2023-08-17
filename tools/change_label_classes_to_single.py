# 该文件是用于将一个数据集中多种类标签改成单种类的标签, 即0.
import os

label_path = r'E:\STUDYCONTENT\Pycharm\yolov5-master\runs\detect\labels'
# label_path = r'D:\StuData\tomato\kaggle\Riped and Unriped tomato Dataset\labels'
txt_list = os.listdir(label_path)
print(txt_list)
for txt_name in txt_list:
    txt_path = os.path.join(label_path, txt_name)
    if txt_name == 'classes.txt':
        continue
    file_txt = open(txt_path, 'r')
    context = file_txt.readlines()
    file_txt.close()
    after_context = ''
    for i in context:
        list_context = i.split()
        list_context[0] = '0' if list_context != '' else None
        before_elem = ' '.join(list_context) + '\n'
        after_context = after_context + before_elem
    print(after_context)
    file_txt = open(txt_path, "w")
    file_txt.write(after_context)
    file_txt.close()
print('done!')