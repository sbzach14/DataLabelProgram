import openpyxl
import json


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
    'NULL':52
}


def CardToIndex(card_name):
    if card_name == 'NULL':
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

if __name__ == "__main__":
    NameList = ['0041']
    for id in NameList:
        excel_path = "Label/OutputLabel/Label"+id+".xlsx"   
        json_path = "Label/"+id+".json"
        merged_dic = Data_Reform(excel_path, json_path)
        # 将字典转换为 JSON 格式的字符串
        json_str = json.dumps(merged_dic)

        # 将 JSON 格式的字符串写入到文件中
        with open("Label/MergedLabel/"+id+"Merged.json", 'w') as f:
            f.write(json_str)
    

