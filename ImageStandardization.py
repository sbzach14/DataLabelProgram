import os
import cv2
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
import Segmentation
from LabelProcess import mask_to_polygon


def ResizeMask(mask, shape):
    resized_mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)

    return resized_mask



def polygon_to_mask(polygon, image_shape):
    # 创建空白掩码图像
    mask = np.zeros(image_shape, dtype=np.uint8)


    for points in polygon:
        # 多边形顶点坐标列表
        points = np.array(points, dtype=np.int32)
        if points.ndim == 1:
            # 将形状从 (n,) 改变为 (1, n)
            points = points.reshape((1, -1))
        # 填充多边形
        cv2.fillPoly(mask, [points], 1)

    if mask.shape != image_shape:
        mask = ResizeMask(mask, image_shape)

    return mask


def ImageStandardizer(folder_path, output_folder):
    
    # os.makedirs(output_folder + '/' + 'Shuffle_'+folder_path[7:], exist_ok=True)
    os.makedirs(output_folder + '/' + folder_path[7:], exist_ok=True)

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)

        img = cv2.imread(image_path)
        if folder_path[-1] == '2':
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        img = cv2.resize(img, (960, 540))
        output_path = os.path.join(output_folder, image_path[7:])
        print(output_path,"  " ,image_path)
        cv2.imwrite(output_path, img)
        

def ImageMaskVerification(imageFolderPath, jsonPath):

    with open(jsonPath,'r') as jsonFile:
        data = json.load(jsonFile)
    
    print(len(os.listdir(imageFolderPath)), "  " , len(data), " " , len(os.listdir(imageFolderPath)) == len(data))

    for imageName in os.listdir(imageFolderPath):
        imagePath = os.path.join(imageFolderPath, imageName)
        img = cv2.imread(imagePath)
        
        segmentationList = data[imageName]['segmentation']

        for segmentPoly in segmentationList:

            mask = polygon_to_mask(segmentPoly, img[:,:,0].shape)

            # data[imageName]['segmentation'].append(mask_to_polygon(mask))

            # Show mask

            plt.figure(figsize=(10,10))
            plt.imshow(img)
            Segmentation.show_mask(mask, plt.gca())
            plt.title(f"Mask" + " " + imageName, fontsize=18)
            plt.axis('off')
            # 等待用户按下任意键
            plt.waitforbuttonpress()
            # 关闭显示窗口
            plt.close()
        
        # del data[imageName]['mask']
        # cate = data[imageName]['categories']
        # del data[imageName]['categories']
        # data[imageName]['Categories'] = cate

        # print(data[imageName])


    # 重写mask label
    # with open('Label/Modified_Cut_To_train_in_Video2.json', 'w') as file:
    #     json.dump(data, file)
            


def RenameJson(jsonFolder):
    for filename in os.listdir(jsonFolder):
        newFileName = 'Shuffle_Label_' + filename[0:4]+".json"
        # 原文件的完整路径
        old_filepath = os.path.join(jsonFolder, filename)
        
        # 新文件的完整路径
        new_filepath = os.path.join(jsonFolder, newFileName)

        # 重命名文件
        os.rename(old_filepath, new_filepath)



  

 


if __name__ == "__main__":
    shuffle_folder_list = ['0022','0024','0025','0027','0028','0030','0033','0034','0035','0037','0041']
    cut_folder_list = ['Modified_Cut_To_train_in_Video1']

    # standardize the image to rotate 90 and resize to 1/2

    # output_folder = "CutImageData"
    # for folder_path in cut_folder_list:
    #     ImageStandardizer('Output/' + folder_path, output_folder)
     

    # verify the image and mask position

    for folderID in shuffle_folder_list:
        if folderID == '0041':
            print(folderID)
            jsonPath = 'Label/MergedLabel/Shuffle_Label_'+ folderID+'.json'
            imageFolderPath = 'ShuffleImageData/Shuffle_' + folderID
            ImageMaskVerification(imageFolderPath, jsonPath)

    # for folderID in cut_folder_list:    
    #     print(folderID)
    #     jsonPath = 'Label/'+folderID+'.json'
    #     imageFolderPath = 'CutImageData/' + folderID[9:]
    #     ImageMaskVerification(imageFolderPath, jsonPath)


    # RenameJson('Label/MergedLabel/')


