# 该文件是用于将模型的opt配置文件解析并且写入当前目录文件夹下的一个表格中, 可以将要解析的opt参数自行添加删除.
import xlwt
import yaml


# cfg, weights, epochs, batch_size, imgsz, multi_scale, optimizer, workers, cos_lr, freeze, label_smoothing, quad
def read_yaml_all(yaml_path):
    try:
        # 打开文件
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            return data
    except:
        return None


# 3.2.2 使用xlwt创建新表格并写入
def write_in(keys, values):
    # 创建新的workbook（其实就是创建新的excel）
    workbook = xlwt.Workbook(encoding='utf-8')

    # 创建新的sheet表
    worksheet = workbook.add_sheet("001")

    # 设置样式
    style = xlwt.XFStyle()
    al = xlwt.Alignment()
    # VERT_TOP = 0x00       上端对齐
    # VERT_CENTER = 0x01    居中对齐（垂直方向上）
    # VERT_BOTTOM = 0x02    低端对齐
    # HORZ_LEFT = 0x01      左端对齐
    # HORZ_CENTER = 0x02    居中对齐（水平方向上）
    # HORZ_RIGHT = 0x03     右端对齐
    al.horz = 0x03  # 设置水平居中
    al.vert = 0x01  # 设置垂直居中
    style.alignment = al

    # 往表格写入内容
    for index in range(len(keys)):
        worksheet.write(0, index, keys[index], style)
        worksheet.write(1, index, str(values[index]), style)

    # 保存
    workbook.save("opt.xls")


if __name__ == '__main__':
    yaml_path_main = \
        r'E:\STUDYCONTENT\Pycharm\yolov5-master\runs\train\5mCBAM_w5m_hypHi_Multi_1_5+8-coco\opt.yaml'

    data_yaml = read_yaml_all(yaml_path_main)
    keys_main = ['cfg', 'weights', 'epochs', 'batch_size', 'imgsz', 'multi_scale', 'optimizer', 'workers', 'cos_lr', 'freeze', 'label_smoothing', 'quad']
    values_main = list(map(data_yaml.get, keys_main))
    # print(values_main)

    write_in(keys_main, values_main)

    # data_exl = xlrd.open_workbook(exl_path_main)
    # table = data_exl.sheets()[0]  # 通过索引顺序获取
    # print(table)

