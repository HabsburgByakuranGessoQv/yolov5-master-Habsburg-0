import json
import os
from pathlib import Path

def coco_to_yolo(coco_file, output_dir):
    with open(coco_file) as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']

    for image_info in images:
        image_name = image_info['file_name']
        image_width = image_info['width']
        image_height = image_info['height']

        label_path = Path(output_dir) / 'labels' / f"{image_name.split('.')[0]}.txt"
        Path(output_dir) / 'images' / image_name

        with open(label_path, 'w') as label_file:
            for annotation in annotations:
                if annotation['image_id'] == image_info['id']:
                    category_id = annotation['category_id']
                    x, y, width, height = annotation['bbox']
                    x_center = (x + width / 2) / image_width
                    y_center = (y + height / 2) / image_height
                    width = width / image_width
                    height = height / image_height

                    label_file.write(f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        # # Copy image to new directory
        # os.makedirs(Path(output_dir) / 'images', exist_ok=True)
        # os.makedirs(Path(output_dir) / 'labels', exist_ok=True)
        # os.system(f"cp {image_name} {Path(output_dir) / 'images'}")

def extract_classes(coco_file, output_dir):
    output_dir = os.path.join(output_dir, 'labels')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(output_dir)
    with open(coco_file) as f:
        data = json.load(f)

    categories = data['categories']
    class_names = [category['name'] for category in categories]

    with open(Path(output_dir) / 'classes.txt', 'w') as classes_file:
        classes_file.write('\n'.join(class_names))

if __name__ == "__main__":
    coco_file_path = r'D:\StuData\tomato\paddle\tomatoOD\tomatOD_annotations\tomatOD_train.json'  # 修改为你的COCO标注文件的路径
    output_directory = r"D:\StuData\tomato\paddle\summary"  # 修改为输出的YOLO数据集目录
    # extract_classes(coco_file_path, output_directory)
    coco_to_yolo(coco_file_path, output_directory)
