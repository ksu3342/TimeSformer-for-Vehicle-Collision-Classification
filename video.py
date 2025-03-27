
from timesformer.models.vit import TimeSformer
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# 设置GPU相关配置
class VideoDataset(Dataset):
    def __init__(self, root_dir, frame_size=(256, 256), num_frames=100):
        # 视频文件根目录
        self.root_dir = root_dir
        # 调整视频帧大小
        self.frame_size = frame_size
        # 每个视频采样的帧数
        self.num_frames = num_frames

        self.video_files = []
        self.labels = []
        # 添加一个权重列表,用于加权采样
        self.weights = []
        for label, category in enumerate(os.listdir(root_dir)):
            category_dir = os.path.join(root_dir, category)
            if os.path.isdir(category_dir):
                files = [os.path.join(category_dir, f) for f in os.listdir(category_dir) if f.endswith('.mp4')]
                self.video_files.extend(files)
                self.labels.extend([label] * len(files))
                # 假设的负样本标签为0,给予更高权重
                self.weights.extend([5 if label == 0 else 1 for _ in files])

        # 转换成torch tensor
        self.weights = torch.tensor(self.weights, dtype=torch.float)

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        # 读取视频帧,调整大小,采样定长帧数
        cap = cv2.VideoCapture(self.video_files[idx])
        frames = []
        for _ in range(self.num_frames):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, self.frame_size)
            frames.append(frame)
        # 如果帧数不够,用0填充
        while len(frames) < self.num_frames:
            frames.append(np.zeros((*self.frame_size, 3)))

        cap.release()

        return torch.from_numpy(np.array(frames)).float(), self.labels[idx]


    # Rest of the code remains unchanged
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
# device=torch.device("cpu")
dataset = VideoDataset('training')
# 创建一个 WeightedRandomSampler
sampler = WeightedRandomSampler(weights=dataset.weights, num_samples=len(dataset), replacement=True)
# 将sampler传给DataLoader
loader = DataLoader(dataset, batch_size=1, sampler=sampler)

model = TimeSformer(img_size=224, num_classes=2, num_frames=100, attention_type='divided_space_time',  pretrained_model='Timesformer/TimeSformer_divST_96x4_224_K400.pyth')
model = model.to(device)
# dummy_video = torch.randn(16, 3, 8, 224, 224) # (batch x channels x frames x height x width)
# dummy_video = dummy_video.to(device)
# pred = model(dummy_video,) # (16, 2)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# 训练循环
for epoch in range(10):
    model.train()
    for i, (videos, labels) in enumerate(loader):
        videos = videos.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # loader输出frames的shape是(N, T, H, W, C)
        frames = videos

        # 1. 调换维度到 (N, C, T, H, W)
        frames = frames.permute(0, 4, 1, 2, 3)

        # # 2. resize到模型设定的图像size,这里是224
        # frames = F.interpolate(frames, size=224)

        # 3. normalize到[0, 1]区间        if i % 10 == 0:
        #             print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}")
        frames = frames / 255.0
        frames = frames - frames.min()
        frames = frames / frames.max()

        # 4. 转换为torch Tensor
        frames = frames.float()
        # 5. 将frames传入TimeSformer模型
        preds = model(frames)

        # 计算损失
        loss = criterion(preds, labels)

        # 反向传播,优化参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}")
    # 评估准确率等指标