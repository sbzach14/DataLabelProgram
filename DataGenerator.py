import os
import cv2

# 设置视频文件夹路径
video_folder = "Video/"

# 获取所有视频文件的文件名
video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]

# 设置输出图片文件夹路径
output_folder = "Output/"

# 循环处理每个视频文件
for video_file in video_files:
    # 获取视频名称（不包含后缀）
    video_name = os.path.splitext(video_file)[0]
    
    # 创建以视频名称命名的文件夹
    output_path = os.path.join(output_folder, video_name)
    os.makedirs(output_path, exist_ok=True)
    
    # 使用FFmpeg打开视频文件
    cap = cv2.VideoCapture(os.path.join(video_folder, video_file))
    
    # 循环读取视频的每一帧
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 保存当前帧为图片文件
        frame_path = os.path.join(output_path, f"{frame_count:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        
        frame_count += 1
    
    # 关闭视频文件
    cap.release()
