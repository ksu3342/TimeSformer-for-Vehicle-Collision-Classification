import cv2
import numpy as np
import os


# 定义数据集类
def __init__(self, video_folder_path, num_frames_per_video, frame_size, label):
        self.video_folder_path = video_folder_path
        self.num_frames_per_video = num_frames_per_video
        self.frame_size = frame_size
        self.label = label
        
#返回数据集的大小
def __len__(self):
        return len(self.video_files)

#视频帧的读取和预处理
def __getitem__(self, index):
        video_file = self.video_files[index]
        video_path = os.path.join(self.video_folder_path, video_file)

# 指定每个视频要读取的帧数,每个帧的尺寸
num_frames_per_video = 100
frame_size = (256, 256)

# 创建一个空的列表来存储所有视频的帧和标签
All_videos = []
All_labels = []

# 定义一个函数来读取视频并返回numpy数组
def read_videos(video_folder_path, label):
    # 获取文件夹中所有视频的文件名
    video_files = [f for f in os.listdir(video_folder_path) if f.endswith('.mp4')]
    
    # 遍历所有的视频文件
    for video_file in video_files:
        # 读取视频文件
        cap = cv2.VideoCapture(os.path.join(video_folder_path, video_file))

        for i in range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), num_frames_per_video):
            # 创建一个空的列表来存储该视频的帧
            frames = []

            # 读取指定数量的帧
            for _ in range(num_frames_per_video):
                # 读取下一帧
                ret, frame = cap.read()

                # 如果帧读取正确，ret将为True
                if not ret:
                    break

                # 将帧resize到指定的尺寸
                frame = cv2.resize(frame, frame_size)

                # 将帧添加到列表中
                frames.append(frame)

            # 如果视频的帧数不足，用零填充
            while len(frames) < num_frames_per_video:
                frames.append(np.zeros((frame_size[0], frame_size[1], 3)))

            # 将帧列表和对应的标签添加到所有视频的列表中
            All_videos.append(frames)
            All_labels.append(label)

        # 释放视频文件
        cap.release()
# 读取negative类别的视频
read_videos('/home/sunyvbo/demo1/dataset/test/n', 0)
read_videos('/home/sunyvbo/demo1/dataset/test/n', 0)

# 读取positive类别的视频
read_videos('/home/sunyvbo/demo1/dataset/test/p', 1)
read_videos('/home/sunyvbo/demo1/dataset/test/p', 1)

# 将帧列表和标签列表转换为numpy数组
All_videos = np.array(All_videos)
All_labels = np.array(All_labels)

print('All Label Shape:', All_labels.shape)
print('All Frames Shape:', All_videos.shape)

# video_folder_path = '/home/sunyvbo/demo1/dataset/test/n'
video_files1 = [f for f in os.listdir('/home/sunyvbo/demo1/dataset/test/n') if f.endswith('.mp4')]
print('Number of  negative video files:', len(video_files1))

# video_folder_path = '/home/sunyvbo/demo1/dataset/test/p'
video_files2 = [f for f in os.listdir('/home/sunyvbo/demo1/dataset/test/p') if f.endswith('.mp4')]
print('Number of  positive video files:', len(video_files2))

print('Total Number of videos files:',len(video_files1)+len(video_files2))

        



 # import cv2
        # import numpy as np
        # import os
        # import torch
        # from torch.utils.data import Dataset, DataLoader
        #
        # class CarCollisionDataset(Dataset):
        #     def __init__(self, video_folder_path, num_frames_per_video, frame_size, label):
        #         self.video_folder_path = video_folder_path
        #         self.num_frames_per_video = num_frames_per_video
        #         self.frame_size = frame_size
        #         self.label = label
        #         self.video_files = [f for f in os.listdir(video_folder_path) if f.endswith('.mp4')]
        #
        #     def __len__(self):
        #         return len(self.video_files)
        #
        #     def __getitem__(self, index):
        #         video_file = self.video_files[index]
        #         video_path = os.path.join(self.video_folder_path, video_file)
        #
        #         # 读取视频文件
        #         cap = cv2.VideoCapture(video_path)
        #
        #         frames = []
        #         for i in range(self.num_frames_per_video):
        #             # 读取下一帧
        #             ret, frame = cap.read()
        #
        #             # 如果帧读取正确，ret将为True
        #             if not ret:
        #                 break
        #
        #             # 将帧resize到指定的尺寸
        #             frame = cv2.resize(frame, self.frame_size)
        #
        #             # 将帧添加到列表中
        #             frames.append(frame)
        #
        #         # 如果视频的帧数不足，用零填充
        #         while len(frames) < self.num_frames_per_video:
        #             frames.append(np.zeros((self.frame_size[0], self.frame_size[1], 3)))
        #
        #         # 释放视频文件
        #         cap.release()
        #
        #         # 转换为NumPy数组
        #         frames = np.array(frames)
        #         label = self.label
        #
        #         return frames, label
        #
        # # 定义数据集和数据加载器
        # train_dataset = CarCollisionDataset('train/negative', num_frames_per_video, frame_size, label=0)
        # train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        #
        # test_dataset = CarCollisionDataset('test/negative', num_frames_per_video, frame_size, label=0)
        # test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

