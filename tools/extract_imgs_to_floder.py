import os
import random
import shutil


def extract_images(source_dir, target_dir, num_images=20):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)

        if os.path.isdir(folder_path):
            target_subdir = os.path.join(target_dir, folder_name)
            # os.makedirs(target_subdir)

            images = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]
            num_images_to_copy = min(num_images, len(images))

            for i in range(num_images_to_copy):
                final_name = '{0}_{1}'.format(folder_name, i + 1)
                source_file = os.path.join(folder_path, images[i])
                target_file = os.path.join(target_dir, f"{final_name}.jpg")
                shutil.copyfile(source_file, target_file)


def extract_random_images(source_dir, target_dir, num_images=20):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)

        if os.path.isdir(folder_path):
            # target_subdir = os.path.join(target_dir, folder_name)
            # if not os.path.exists(target_subdir):
            #     os.makedirs(target_subdir)

            images = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]
            num_images_to_copy = min(num_images, len(images))

            # Randomly sample 'num_images_to_copy' images from the list
            sampled_images = random.sample(images, num_images_to_copy)

            for i, image in enumerate(sampled_images):
                final_name = '{0}_{1}'.format(folder_name, i + 1)
                source_file = os.path.join(folder_path, image)
                target_file = os.path.join(target_dir, f"{final_name}.jpg")
                shutil.copyfile(source_file, target_file)


if __name__ == "__main__":
    source_directory = r"D:\StuData\tomato\kaggle\Ball"  # 替换为你的源文件夹路径
    target_directory = r"D:\StuData\tomato\dataset_factory\temp\ball\images"  # 替换为你的目标文件夹路径
    num_main = 270
    # extract_images(source_directory, target_directory, num_images=num_main)
    extract_random_images(source_directory, target_directory, num_images=num_main)
