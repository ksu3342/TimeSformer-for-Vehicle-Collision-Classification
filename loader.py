import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader

# 定义数据集类
class CarCollisionDataset(Dataset):
    def __init__(self, video_folder_path, num_frames_per_video, frame_size, label):
        self.video_folder_path = video_folder_path
        self.num_frames_per_video = num_frames_per_video
        self.frame_size = frame_size
        self.label = label
        self.video_files = [f for f in os.listdir(video_folder_path) if f.endswith('.mp4')]
    #返回数据集的大小
    def __len__(self):
        return len(self.video_files)
    #视频帧的读取和预处理
    def __getitem__(self, index):
        video_file = self.video_files[index]
        video_path = os.path.join(self.video_folder_path, video_file)

        # 读取视频文件
        cap = cv2.VideoCapture(video_path)

        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频总帧数

        # 计算步长，确保选择的帧数为100
        step = max(total_frames // self.num_frames_per_video, 1)

        # 选择指定帧数的视频帧
        for i in range(0, total_frames, step):
            # 设置当前帧位置
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)

            # 读取帧
            ret, frame = cap.read()

            # 如果帧读取正确，ret将为True
            if not ret:
                break

            # 将帧resize到指定的尺寸
            frame = cv2.resize(frame, self.frame_size)

            # 将帧添加到列表中
            frames.append(frame)

            # 如果已经选择了足够的帧数，停止循环
            if len(frames) == self.num_frames_per_video:
                break

        # 如果视频的帧数不足，用零填充
        while len(frames) < self.num_frames_per_video:
            frames.append(np.zeros((self.frame_size[0], self.frame_size[1], 3)))

        # 释放视频文件
        cap.release()

        # 转换为NumPy数组
        frames = np.array(frames)
        label = self.label

        # 打印帧列表和标签列表
        print('Frames:', frames.shape)
        print('Label:', label)
        return frames, label


# 定义数据集和数据加载器
num_frames_per_video = 100  
frame_size = (256, 256)  

train_dataset = CarCollisionDataset('C:\Users\20162\Desktop\CarCollisionDataset', num_frames_per_video, frame_size, label=1)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = CarCollisionDataset('C:\Users\20162\Desktop\CarCollisionDataset', num_frames_per_video, frame_size, label=0)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
