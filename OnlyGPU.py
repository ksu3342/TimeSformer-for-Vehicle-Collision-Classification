import cv2 
import torch
from timesformer.models.vit import TimeSformer
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler 
import os
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import matplotlib.pyplot as plt


# 设置GPU相关配置
torch.backends.cuda.matmul.allow_tf32 = False  
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
device = torch.device("cuda:3") 
# 设置GPU设备 


# 初始化一个列表保存每次迭代的loss值
losses = []
# 定义读取和预处理视频数据的Dataset类
class VideoDataset(Dataset):
    def __init__(self, root_dir, frame_size=(224, 224), num_frames=100):
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


# 检查是否有可用GPU 
print(torch.cuda.is_available())
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# 构建数据集
dataset = VideoDataset('/home/sunyvbo/TEXT')  
# 创建一个 WeightedRandomSampler，并进行权重的检查
sampler = WeightedRandomSampler(weights=dataset.weights, num_samples=len(dataset), replacement=True)
assert torch.all(dataset.weights >= 0), "Weights should be non-negative."
assert torch.all(dataset.weights > 0), "At least one weight should be positive."

print(len(dataset))

# 将sampler传给DataLoader
loader = DataLoader(dataset, batch_size=1, sampler=sampler) 

# 构建模型,定义优化器
model = TimeSformer(img_size=224, num_classes=2, num_frames=100, attention_type='divided_space_time',  pretrained_model='/home/sunyvbo/demo1/dataset/TimeSformer-main/Timesformer/TimeSformer_divST_96x4_224_K600.pyth')

model = model.to(device)
# dummy_video = torch.randn(4, 3, 100, 224, 224) # (batch x channels x frames x height x width)

# pred = model(dummy_video,) # (2, 400)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
loss_fn = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss()

# 训练循环
for epoch in range(10):

    model.train()  
    total_loss = 0.0
    num_batches = 0

    tp, tn, fp, fn = 0, 0, 0, 0 # TP, TN, FP, FN
    
    for i, (videos, labels) in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):

        # 发送数据到设备
        videos = videos.to(device) 
        labels = labels.to(device)
        
        optimizer.zero_grad()

        # 前向传播,计算预测值
        frames = videos.permute(0, 4, 1, 2, 3) 
        frames = frames / 255.0
        frames = frames.float()
        preds = model(frames)

        # 计算Loss
        loss = criterion(preds, labels)

        # 更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计TP/TN/FP/FN
        preds = torch.argmax(preds, dim=1)
        for j in range(preds.shape[0]):
            if labels[j] == 1:
                if preds[j] == 1:
                    tp += 1
                else:
                    fn += 1
            else:
                if preds[j] == 0:
                    tn += 1
                else:
                    fp += 1

        # 累加Loss和Batch数
        total_loss += loss.item()
        num_batches += 1

        # 输出指标
        precision = tp / (tp + fp)
        recall = tp / (tp + fn) 
        f1 = 2 * precision * recall / (precision + recall)
        accuracy = (tp + tn) / (tp + fn + fp + tn)

        print(f"Epoch: {epoch}, Loss: {loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")

    # 输出平均Loss 
    avg_loss = total_loss / num_batches
    print(f"Epoch: {epoch}, Average Loss: {avg_loss}")

    # 释放GPU内存
    torch.cuda.empty_cache()

print(len(losses))
plt.figure()
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss') 
plt.title('Training Loss')
# plt.show()
plt.savefig('/home/sunyvbo/demo1/dataset/CODE/figs/savefig_example.png')

