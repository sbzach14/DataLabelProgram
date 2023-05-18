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
from DataReformer import CardToIndex
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
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    # 调整图片大小
    # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
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
    

    
    polygons =  mask_to_polygon(masks)
    return polygons


def EnterLabelAndDump():
    card_name = input('Category:')
    return CardToIndex(card_name)


if __name__ == "__main__":   
    # test code
    # polygon = imageLabeler("testSample.jpg") 

    # true code
    folder_path = "Output/Cut_To_train_in_Video2"
    shuffle_threshold = 10000
    # data = {imageID: [polygonL, polygonR]}
    data = {}
    # Traverse
    count = 0
    for filename in os.listdir(folder_path):
        if count > shuffle_threshold:
            break
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # left and right
            print("Picture Now ",filename, "   " + str(count))
            polygon_list = []
            for i in range(1):
                # 构造图像文件的相对路径
                image_path = os.path.join(folder_path, filename)
                # 调用函数B处理图像
                polygon = imageLabeler(image_path)
                polygon_list.append(polygon)
            data[filename] = {}
            data[filename]['segmentation'] = polygon_list
            data[filename]['Categories'] = [EnterLabelAndDump()]
            print(data[filename])   
            count = count + 1
    with open("NewCutting2.json", "w") as out:
        json.dump(data, out)
    

        
    
            

    
    

