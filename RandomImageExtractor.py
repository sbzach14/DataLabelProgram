import os
import random
import shutil
import string


# 指定输入文件夹路径和输出文件夹路径
input_folder = "IdleVideo"
output_folder = "ActionVideo/FormerIdle"

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹中以"idle"开头的文件夹
for root, dirs, files in os.walk(input_folder):
    for directory in dirs:
        if directory.startswith("IMG"):
            current_folder = os.path.join(root, directory)
            image_files = [f for f in os.listdir(current_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
            # 随机选择不重复的100张图片
            selected_images = random.sample(image_files, len(image_files) // 10)

            # 复制选中的图片到输出文件夹
            for image in selected_images:
                src_path = os.path.join(current_folder, image)
                extension = os.path.splitext(image)[1]
                # 生成新的文件名
                new_filename = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
                dst_path = os.path.join(output_folder, new_filename + extension)
                shutil.copy2(src_path, dst_path)

print("已完成图片选择和复制。")