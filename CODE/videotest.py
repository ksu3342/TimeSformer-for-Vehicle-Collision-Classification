import cv2
import os

def print_video_info(video_folder_path):
    # 获取文件夹中所有视频的文件名
    video_files = [f for f in os.listdir(video_folder_path) if f.endswith('.mp4')]

    i=0
    # 遍历所有的视频文件
    for video_file in video_files:
        # 读取视频文件
        cap = cv2.VideoCapture(os.path.join(video_folder_path, video_file))

        # 获取并打印帧数、宽度和高度
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f'Video {video_file}:')
        print(f'Frame count: {frame_count}')
        print(f'Frame width: {frame_width}')
        print(f'Frame height: {frame_height}')

        # 释放视频文件
        cap.release()
        i=i+1
        print(i)
# 使用你的视频文件夹路径
print_video_info('D:\20162\Videos')