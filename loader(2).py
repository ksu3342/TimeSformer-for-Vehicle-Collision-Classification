# 导入PyTorch的数据加载模块
import cv2 
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler 
import os
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

# 设置GPU相关配置
torch.backends.cuda.matmul.allow_tf32 = False  
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
device = torch.device("cuda:4") 
# 设置GPU设备 
torch.cuda.set_device(device)

# 定义读取和预处理视频数据的Dataset类
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
                # 假设负样本的标签为0,给予更高权重
                self.weights.extend([1 if label == 0 else 5 for _ in files])

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

# 定义视频分类Transformer模型
class VideoTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, num_classes):
        super(VideoTransformer, self).__init__()

        self.embed_dim = embed_dim

        # 线性层将原始视频帧映射到embed_dim维
        self.embedding = nn.Linear(256*256*3, embed_dim)  

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类头
        self.classifier = nn.Linear(embed_dim, num_classes)   

    def forward(self, x):
        # 重新塑形帧以输入到线性层
        x = x.view(x.shape[0], x.shape[1], -1)  
        x = self.embedding(x)  
        # 在时间维度上输入到Transformer
        x = self.transformer(x.permute(1, 0, 2))  
        # 在时间维度上求平均
        x = x.mean(dim=0)
        x = self.classifier(x)
        return x

# 检查是否有可用GPU 
print(torch.cuda.is_available())
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

# 构建数据集
dataset = VideoDataset('/home/sunyvbo/TEXT')  
# 创建一个 WeightedRandomSampler
sampler = WeightedRandomSampler(weights=dataset.weights, num_samples=len(dataset), replacement=True)
print(len(dataset))
# 将sampler传给DataLoader
loader = DataLoader(dataset, batch_size=16, sampler=sampler) 

# 构建模型,定义优化器
model = VideoTransformer(embed_dim=256, num_heads=8, num_layers=3, num_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
loss_fn = nn.CrossEntropyLoss()

# 训练循环
num_epochs = 2
for epoch in range(num_epochs):
    for i, (videos, labels) in enumerate(loader):
        # 发送数据到设备
        videos = videos.to(device)
        labels = labels.to(device)

        # 梯度清零
        optimizer.zero_grad()
            
        # 前向传播
        outputs = model(videos)  

        # 计算损失 
        loss = loss_fn(outputs, labels)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}")
        # Clear GPU memory
        # torch.cuda.empty_cache()
    torch.save(model.state_dict(), 'model_weights.pth') # 保存模型
# 初始化loss数组
losses = []

for epoch in range(num_epochs):
    for i, (videos, labels) in enumerate(loader):
      
        # ...训练代码
        
        if i % 10 == 0:
            losses.append(loss.item())

# 绘制loss曲线  
plt.figure()
plt.plot(losses)
plt.xlabel('Iteration')  
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()