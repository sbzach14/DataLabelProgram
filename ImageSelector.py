import os
import matplotlib.pyplot as plt

# 设置文件夹路径和输出文件夹路径
input_folder = 'OutPut/Cuting2'
output_folder = 'OutPut/NewCutting2'

# 获取文件夹中的所有文件名
files = os.listdir(input_folder)

# 获取文件名列表中的所有图像文件名
image_files = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]

# 循环显示每个图像
current_index = 0
def on_key_press(event):
    global current_index
    print('Key pressed:', event.key)
    if event.key == 'left':
        # 执行左键操作
        # 获取输出文件名
        output_filename = os.path.join(output_folder, image_files[current_index])
        # 保存图像
        plt.imsave(output_filename, img)
        print(f"Saved image to {output_filename}")
        # 显示下一张图像
        current_index = (current_index + 1) % len(image_files)
        plt.close()
    elif event.key == 'right':
        # 执行右键操作
        current_index = (current_index + 1) % len(image_files)
        plt.close()

while True:
    # 读取图像文件
    filename = os.path.join(input_folder, image_files[current_index])
    img = plt.imread(filename)
    # 显示图像
    plt.imshow(img)
    plt.title(f"{current_index+1}/{len(image_files)}: {filename}")
    # 注册事件监听函数
    plt.gcf().canvas.mpl_connect('key_press_event', on_key_press)
    # 显示窗口并等待事件发生
    plt.show(block=True)
