import os
import random
import shutil
from tqdm import tqdm

def copy_random_images(source_folder, destination_folder, num_images):
    # 获取源文件夹中所有图片的列表
    image_files = [f for f in os.listdir(source_folder) if f.endswith('.jpg') or f.endswith('.png')]

    # 过滤出没有 "_A" 后缀的图片
    eligible_images = [image for image in image_files if not image.endswith('_A.jpg') and not image.endswith('_A.png')]

    # 从符合条件的图片列表中随机选择指定数量的图片
    selected_images = random.sample(eligible_images, num_images)

    # 复制选定的图片到目标文件夹并重命名
    for image in tqdm(selected_images):
        source_path = os.path.join(source_folder, image)
        destination_path = os.path.join(destination_folder, image)
        
        # 重命名图片
        name, ext = os.path.splitext(image)
        new_name = name + '_A' + ext
        destination_path_renamed = os.path.join(destination_folder, new_name)
        path_renamed = os.path.join(source_folder, new_name)
        shutil.copy2(source_path, destination_path_renamed)
        os.rename(source_path,path_renamed)

# 调用函数进行复制
source_folder = r'./unlabeled2017'
destination_folder = r'E:/datasets'
num_images = 23000
copy_random_images(source_folder, destination_folder, num_images)

