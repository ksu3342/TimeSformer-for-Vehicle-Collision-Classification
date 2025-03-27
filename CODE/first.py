import cv2 
import torch
from torch import nn, einsum, optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler 
import os
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# 设置GPU相关配置
torch.backends.cuda.matmul.allow_tf32 = False  
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
device = torch.device("cuda:4") 
# 设置GPU设备 
torch.cuda.set_device(device)

# 初始化一个列表保存每次迭代的loss值
losses = []
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
                # 假设正样本的标签为1,给予更高权重
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

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden_dim, out_dim),nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, num_patches_space=None, num_patches_time=None, attn_type=None):
        super().__init__()

        assert attn_type in ['space', 'time'], 'Attention type should be one of the following: space, time.'

        self.attn_type = attn_type
        self.num_patches_space = num_patches_space
        self.num_patches_time = num_patches_time

        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

    def forward(self, x):

        t = self.num_patches_time
        n = self.num_patches_space

        # reshape to reveal dimensions of space and time
        x = rearrange(x, 'b (t n) d -> b t n d', t=t, n=n)

        if self.attn_type == 'space':
            out = self.forward_space(x) # (b, tn, d)
        elif self.attn_type == 'time':
            out = self.forward_time(x) # (b, tn, d)
        else:
            raise Exception('Unknown attention type: %s' % (self.attn_type))

        return out

    def forward_space(self, x):
        """
        x: (b, t, n, d)
        """

        t = self.num_patches_time
        n = self.num_patches_space

        # hide time dimension into batch dimension
        x = rearrange(x, 'b t n d -> (b t) n d')  # (bt, n, d)

        # apply self-attention
        out = self.forward_attention(x)  # (bt, n, d)

        # recover time dimension and merge it into space
        out = rearrange(out, '(b t) n d -> b (t n) d', t=t, n=n)  # (b, tn, d)

        return out

    def forward_time(self, x):
        """
        x: (b, t, n, d)
        """

        t = self.num_patches_time
        n = self.num_patches_space

        # hide time dimension into batch dimension
        x = x.permute(0, 2, 1, 3)  # (b, n, t, d)
        x = rearrange(x, 'b n t d -> (b n) t d')  # (bn, t, d)

        # apply self-attention
        out = self.forward_attention(x)  # (bn, t, d)

        # recover time dimension and merge it into space
        out = rearrange(out, '(b n) t d -> b (t n) d', t=t, n=n)  # (b, tn, d)

        return out

    def forward_attention(self, x):
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return out


class Transformer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout, num_patches_space, num_patches_time):
        super().__init__()

        self.num_patches_space = num_patches_space
        self.num_patches_time = num_patches_time
        heads_half = int(heads / 2.0)

        assert dim % 2 == 0

        self.attention_space = PreNorm(dim, Attention(dim, heads=heads_half, dim_head=dim_head, dropout=dropout, num_patches_space=num_patches_space, num_patches_time=num_patches_time, attn_type='space'))
        self.attention_time = PreNorm(dim, Attention(dim, heads=heads_half, dim_head=dim_head, dropout=dropout, num_patches_space=num_patches_space, num_patches_time=num_patches_time, attn_type='time'))

        inner_dim = dim_head * heads_half * 2
        self.linear = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.mlp = PreNorm(dim, FeedForward(dim, mlp_dim, dim, dropout=dropout))

    def forward(self, x):

        # self-attention
        xs = self.attention_space(x)
        xt = self.attention_time(x)
        out_att = torch.cat([xs, xt], dim=2)

        # linear after self-attention
        out_att = self.linear(out_att)

        # residual connection for self-attention
        out_att += x

        # mlp after attention
        out_mlp = self.mlp(out_att)

        # residual for mlp
        out_mlp += out_att

        return out_mlp


class ViViT(nn.Module):
    def __init__(self, num_classes, clip_size):
        super().__init__()

        self.image_size = 256
        self.patch_size = 28
        self.num_classes = 2
        self.clip_size = 100
        self.pool_type = 'cls'
        self.dim = 256
        self.depth = 6
        self.heads = 16
        self.mlp_dim = 256
        self.channels = 3
        self.dim_head = 64
        self.dropout_ratio = 0.1
        self.emb_dropout_ratio = 0.1

        assert self.heads % 2 == 0, 'Number of heads should be even.'

        assert self.image_size % self.patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        self.num_patches_time = clip_size
        self.num_patches_space = (self.image_size // self.patch_size) ** 2
        self.num_patches = self.num_patches_time * self.num_patches_space

        self.patch_dim = self.channels * self.patch_size ** 2
        assert self.pool_type in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # init layers of the classifier
        self._init_layers()

        # init loss, metric, and optimizer
        self._loss_fn = nn.CrossEntropyLoss()
        self._metric_fn = metrics.accuracy
        # self._optimizer = optim.Adam(self.parameters(), 0.001)
        self._optimizer = optim.SGD(self.parameters(), 0.1)

    def _init_layers(self):

        self.to_patch_embedding = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size), nn.Linear(self.patch_dim, self.dim), )
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, self.dim))
        self.dropout = nn.Dropout(self.emb_dropout_ratio)

        self.transformers = nn.ModuleList([])
        for _ in range(self.depth):
            self.transformers.append(Transformer(self.dim, self.heads, self.dim_head, self.mlp_dim, self.dropout_ratio, self.num_patches_space, self.num_patches_time))

        self.mlp_head = nn.Sequential(nn.LayerNorm(self.dim), nn.Linear(self.dim, self.num_classes), nn.Softmax())

    def forward(self, x):

        b, c, t, h, w = x.shape

        # hide time inside batch
        x = x.permute(0, 2, 1, 3, 4)  # (b, t, c, h, w)
        x = rearrange(x, 'b t c h w -> (b t) c h w')  # (b*t, c, h, w)

        # input embedding to get patch token
        x = self.to_patch_embedding(x)  # (b*t, n, d)

        # concat patch token and class token
        x = rearrange(x, '(b t) n d -> b (t n) d', b=b, t=t)  # (b, tn, d)

        # add position embedding
        x += self.pos_embedding  # (b, tn, d)
        x = self.dropout(x)  # (b, tn, d)

        # layers of transformers
        for transformer in self.transformers:
            x = transformer(x)  # (b, tn, d)

        # space-time pooling
        x = x.mean(dim=1)

        # classification
        # x = x[:, 0]

        # classifier
        x = self.mlp_head(x)
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
loader = DataLoader(dataset, batch_size=4, sampler=sampler) 

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
            # 在训练循环中,将每个batch的loss添加到列表中
            losses.append(loss.item())

        # Clear GPU memory
        # torch.cuda.empty_cache()
    # 保存模型
    torch.save(model.state_dict(), 'model_weights.pth') 
# 训练完成后,使用Matplotlib绘制loss曲线
print(len(losses))
plt.figure()
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss') 
plt.title('Training Loss')
plt.show()
