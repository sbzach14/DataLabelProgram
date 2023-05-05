import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
import Segmentation

# 定义回调函数
points = []
def get_mouse_position(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print("Point selected:", (x, y))

# 读取sam model
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
# device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device=device)
print("Load model successfully")

# 读取图片
img = cv2.imread('OriginalSample1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

# 调整图片大小
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
# 显示图片
cv2.imshow('image', img)

# 设置鼠标回调函数
cv2.setMouseCallback('image', get_mouse_position)

# 等待选点
while True:
    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == ord('q') or len(points) == 2:
        break

# 关闭窗口
cv2.destroyAllWindows()

# 输出点的位置
print("Selected points:", points)


predictor = SamPredictor(sam)
predictor.set_image(img)


input_point = np.array(points)
input_label = np.array([1,1])

plt.figure(figsize=(10,10))
plt.imshow(img)
Segmentation.show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.show()

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False,
)

print("mask shape: ", masks.shape, " masks: ", masks)

for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    Segmentation.show_mask(mask, plt.gca())
    Segmentation.show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()  
