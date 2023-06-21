import openpyxl
import json

import numpy as np
from pycocotools import mask as mask_utils
import os
import cv2

CardDic = {
    'A':0,
    'B':1,
    'C':2,
    'D':3,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    '0':10,
    'J':11,
    'Q':12,
    'K':13,
    'NULL':52,
    'BLACKJOKER':53,
    'REDJOKER':54
}

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



def CardToIndex(card_name):
    if card_name == 'NULL' or card_name == 'REDJOKER' or card_name == 'BLACKJOKER':
        return CardDic[card_name]
    else:
        return (CardDic[card_name[0]] * 13 +CardDic[card_name[1]] - 1) 

def Data_Reform(excel_path, json_path):
    with open(json_path, 'r') as jsonfile:
        mask_data = json.load(jsonfile)
        # 打开Excel文件
        workbook = openpyxl.load_workbook(excel_path)

        # 选择要读取的工作表
        sheet = workbook['工作表1']

        # 创建一个空字典
        B = {}
        # 从第二行开始遍历工作表
        for row in sheet.iter_rows(min_row=2, values_only=True):
            # 取出图片序号作为键
            imagekey = str(int(row[0])).zfill(6) + '.jpg'
            # 取出mask
            mask = mask_data[imagekey]
            # 创建一个新的字典作为值
            value = {"Categories": [CardToIndex(row[1]),CardToIndex(row[2])], "segmentation" :mask,"isBlur": [int(row[3]),int(row[4])], "RelativePos": [int(row[5]),int(row[6])]}
            # 将键值对添加到B字典中
            B[imagekey] = value
    return B


def encode_rle_for_nparray(mask):
    # 将掩码数组转换为二进制形式
    binary_mask = (mask > 0).astype(np.uint8)

    # 使用 pycocotools.mask.encode 函数进行编码
    rle_encoded = mask_utils.encode(np.asfortranarray(binary_mask))

    return rle_encoded

def encode_rle_for_binaryarray(mask):

    rle_encode = mask_utils.encode(np.asfortranarray(mask))
    
    return rle_encode

if __name__ == "__main__":
    NameList = ['0387','0389','0391','0392','0393','0397']
    for id in NameList:
        print(id)
        excel_path = "Label/OutputLabel/Label"+id+".xlsx"   
        json_path = id+".json"
        merged_dic = Data_Reform(excel_path, json_path)
        # 将字典转换为 JSON 格式的字符串
        json_str = json.dumps(merged_dic)

        # 将 JSON 格式的字符串写入到文件中
        with open("Label/MergedLabel/" + "Shuffle_Label_" + id + ".json", 'w') as f:
            f.write(json_str)

    output_path = "Label/MergedLabel/RLELabel/"
    folder_path = "Label/MergedLabel/"



    # img = cv2.imread("ShuffleImageData/Shuffle_0022/000000.jpg")
    # for filename in os.listdir(folder_path):
    #     if filename.endswith('.json'):
    #         file_path = os.path.join(folder_path + filename)

    #         with open(file_path, 'r') as file:
    #             data = json.load(file)
            
    #         for image_name in data.keys():

    #             segmentList = data[image_name]['segmentation']

    #             for i in range(len(segmentList)):
    #                 segmentPoly = segmentList[i]
    #                 mask = polygon_to_mask(segmentPoly, img[:,:,0].shape)
    #                 rle_encode = encode_rle_for_nparray(mask)
    #                 # 将字节字符串转换为字符串形式
    #                 rle_encode['counts'] = rle_encode['counts'].decode()
    #                 data[image_name]['segmentation'][i] = rle_encode
    #                 print(rle_encode)

            
    #         with open(output_path + filename,'w') as newLabel:
    #             json.dump(data, newLabel)

                





    

