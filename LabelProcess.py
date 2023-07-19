import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
import Segmentation
import os
import json
from DataReformer import CardToIndex, encode_rle_for_binaryarray, encode_rle_for_nparray
# 定义回调函数


def mask_to_polygon(mask):
    mask = np.squeeze(mask)  # 删除多余的维度
    mask = mask.astype(np.uint8) * 255
    # print("mask shape: ", mask.shape, " mask: ", mask, "dataType:", mask.dtype)


    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [np.squeeze(contour).tolist() for contour in contours]
    return polygons

# 读取sam model

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
print("Load model successfully")

def imageLabeler(imagePath):

    points = []
    def get_mouse_position(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print("Point selected:", (x, y))

    # 读取图片
    img = cv2.imread(imagePath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(img.shape)
    # 显示图片
    cv2.imshow('image', img)
    # 设置鼠标回调函数
    cv2.setMouseCallback('image', get_mouse_position)

    # 等待选点
    while True:
        cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == ord('q') or len(points) == 1:
            break
    # 关闭窗口
    cv2.destroyAllWindows()
    # 输出点的位置
    print("Selected points:", points)
    predictor = SamPredictor(sam)
    predictor.set_image(img)
    input_point = np.array(points)
    input_label = np.array([1])
    # plt.figure(figsize=(10,10))
    # plt.imshow(img)
    # Segmentation.show_points(input_point, input_label, plt.gca())
    # plt.axis('on')
    # plt.show()
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    print(masks.shape)
    # print("masks shape: ", masks.shape, " masks: ", masks, "dataType:", masks.dtype)
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(img)
        Segmentation.show_mask(mask, plt.gca())
        Segmentation.show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        # plt.show()
        # 等待用户按下任意键
        plt.waitforbuttonpress()
        # 关闭显示窗口
        plt.close()
    

    # return polygons
    # polygons =  mask_to_polygon(masks)
    masks = np.squeeze(masks)
    rle_code = encode_rle_for_binaryarray(masks)
    return rle_code


def EnterLabelAndDump():
    card_name = input('Category:')
    return CardToIndex(card_name)


def display_images(folder_path):
    image_files = [file for file in os.listdir(folder_path) if file.endswith(('.jpg', '.jpeg', '.png'))]
    total_images = len(image_files)
    current_image_index = 0

    while current_image_index < total_images:
        image_path = os.path.join(folder_path, image_files[current_image_index])
        image = cv2.imread(image_path)
        
 

        cv2.imshow("Image Viewer", image)
        cv2.setWindowTitle("Image Viewer", image_files[current_image_index])  # 设置窗口标题

        key = cv2.waitKey(0)

        if key == ord(' '):  # 按下空格键显示下一张图片
            current_image_index += 1
        elif key == 27:  # 按下ESC键退出循环
            break

    cv2.destroyAllWindows()


def LabelCuttingCategory(folder_path):

    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    class_value = ""
    label_dic = {}
     
    for image_file in image_files:
        # 显示当前图片
        print("当前图片:", image_file, "当前classvalue", class_value)
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)
        cv2.imshow("Image", image)
         # 输入种类
        if class_value:
            input_text = f"输入图片种类 (默认: {class_value}): "
        else:
            input_text = "输入图片种类: "

        while True:
            key = cv2.waitKey(0)
            if key == ord('r') or key == ord('R'):
                user_input = input(input_text)
                if user_input.strip():
                    class_value = user_input
                break
            elif key == 32:  # 空格键
                break

        label_dic[image_file] = {}
        label_dic[image_file]["Categories"] = [CardToIndex(class_value)]
        label_dic[image_file]["segmentation"] = []

    return label_dic




if __name__ == "__main__":   
    #用SAM标注图片mask###########################################
    # true code
    folder_path = "ShuffleImageDataWithJoker/AfterStandardization/0397"
    # Index of the last image
    shuffle_threshold = 115
    # data = {imageID: [polygonL, polygonR]}
    data = {}
    # Traverse
    # count = 0
    # for filename in os.listdir(folder_path):
    #     if count > shuffle_threshold:
    #         break
    #     if filename.endswith(".jpg") or filename.endswith(".png"):
    #         # left and right
    #         print("Picture Now ",filename, "   " + str(count))
    #         RLE_List = []
    #         for i in range(2):
    #             # 构造图像文件的相对路径
    #             image_path = os.path.join(folder_path, filename)
    #             # 调用函数B处理图像
    #             rle_code = imageLabeler(image_path)
    #             rle_code['counts'] = rle_code["counts"].decode()
    #             RLE_List.append(rle_code)
    #         data[filename] = {}
    #         data[filename]['segmentation'] = RLE_List
    #         print(data[filename])   
    #         count = count + 1
    # with open("0397.json", "w") as out:
    #     json.dump(data, out)

    # 显示图片############################################
    

    display_images("Image\Shuffle_0755")
    # LABELED_FOLDER = ["Cutting_1361", "Cutting_1362"]
    # with open("CutVIdeo\Cutting_Label_1363.json", "w") as l:
    #     label_dic = LabelCuttingCategory("CutVIdeo\Cutting_1363")
    #     json.dump(label_dic, l)
    

    
            

    
    

