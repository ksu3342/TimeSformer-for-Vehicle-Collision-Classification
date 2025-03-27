# 导入所需的库
from timesformer.models.vit import TimeSformer
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import os
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# 初始化一个列表保存每次迭代的loss值
losses = []
test_losses = []

# 定义视频数据集类
class VideoDataset(Dataset):
    def __init__(self, root_dir, frame_size=(256, 256), num_frames=100):
        self.root_dir = root_dir
        self.frame_size = frame_size
        self.num_frames = num_frames

        self.video_files = []
        self.labels = []
        self.weights = []

        # 遍历根目录下的所有文件
        for label, category in enumerate(os.listdir(root_dir)):
            category_dir = os.path.join(root_dir, category)
            if os.path.isdir(category_dir):
                files = [os.path.join(category_dir, f) for f in os.listdir(category_dir) if f.endswith('.mp4')]
                self.video_files.extend(files)
                self.labels.extend([label] * len(files))
                self.weights.extend([5 if label == 0 else 1 for _ in files])

        self.weights = torch.tensor(self.weights, dtype=torch.float)

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        # 打开视频文件
        cap = cv2.VideoCapture(self.video_files[idx])
        # 初始化一个空列表用于存储帧
        frames = []
        for _ in range(self.num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            # 调整帧的大小为指定尺寸
            frame = cv2.resize(frame, self.frame_size)
            # 将帧添加到帧列表中
            frames.append(frame)
        # 如果帧数不足，用空帧填充
        while len(frames) < self.num_frames:
            frames.append(np.zeros((*self.frame_size, 3)))

        cap.release()

        # 将帧列表转换为PyTorch的浮点张量，并与相应的标签一起返回
        return torch.from_numpy(np.array(frames)).float(), self.labels[idx]

# 检查是否有可用GPU 
print("GPU是否可用:",torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 创建模型之前，先将默认设备设置为指定GPU
torch.cuda.set_device(0)

# 构建数据集
dataset = VideoDataset('/home/sunyvbo/TEXT')  
print("数据集长度:", len(dataset))

n_folds = 4
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=33)

# 用于存储每个fold的最佳epoch和其对应的F1分数
best_epochs = []
best_f1s = []

# 用于存储所有fold的总体指标
total_test_loss = 0.0
total_recall = 0.0
total_specificity = 0.0
total_f1 = 0.0

# 开始训练和测试
for fold, (train_idx, test_idx) in enumerate(skf.split(dataset.video_files, dataset.labels)):
    print(f"开始交叉验证:Fold {fold + 1}/{n_folds}")

    # 根据交叉验证的索引,从dataset中按比例采样得到训练集、验证集
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    # 构建加权采样器,对训练集和验证集分别进行加权采样
    sampler = WeightedRandomSampler(weights=dataset.weights[train_idx], num_samples=len(train_idx), replacement=True)
    test_sampler = WeightedRandomSampler(weights=dataset.weights[test_idx], num_samples=len(test_idx), replacement=True)

    # 构建训练集和验证集的 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=1*8, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=1*8, sampler=test_sampler)

    model = TimeSformer(img_size=224, num_classes=2, num_frames=100, attention_type='divided_space_time', pretrained_model='/home/sunyvbo/demo1/dataset/TimeSformer-main/Timesformer/TimeSformer_divST_96x4_224_K600.pyth')
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7])
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # 用于存储当前fold的最佳epoch和其对应的F1分数
    best_epoch = -1
    best_f1 = -1

    # 开始每个epoch的训练
    for epoch in range(5):
        model.train()
        total_loss = 0.0
        num_batches = 0

        print(f"训练集加载的视频数量: {len(train_dataset)}")
        for i, (videos, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            videos = videos.to(device = 0)
            labels = labels.to(device = 0)

            frames = videos
            frames = frames.permute(0, 4, 1, 2, 3)
            frames = frames / 255.0
            frames = frames - frames.min()
            frames = frames / frames.max()
            frames = frames.float()

            preds = model(frames)

            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch}, Average Loss: {avg_loss}")
        # 在训练循环中,将每个batch的loss添加到列表中
        losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            fold_test_loss = 0
            fold_recall = 0
            fold_specificity = 0

            all_preds = []
            all_labels = []
            print(f"测试集加载的视频数量: {len(test_dataset)}")

            for i, (videos, labels) in enumerate(tqdm(test_loader, desc=f"Epoch {epoch}")):
                videos = videos.to(device = 0)
                labels = labels.to(device = 0)

                frames = videos
                frames = frames.permute(0, 4, 1, 2, 3)
                frames = frames / 255.0
                frames = frames - frames.min()
                frames = frames / frames.max()
                frames = frames.float()

                preds = model(frames)

                all_preds.extend(preds.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                loss = criterion(preds, labels)

                fold_test_loss = fold_test_loss + loss.item()
                

                recall = recall_score(labels.cpu().numpy(), preds.argmax(1).cpu().numpy(), zero_division=1)
                fold_recall += recall

                 # 计算混淆矩阵
                confusion_mat = confusion_matrix(labels.cpu().numpy(), preds.argmax(1).cpu().numpy())
                if confusion_mat.shape == (2, 2):
                    tn, fp, fn, tp = confusion_mat.ravel()
                else:
                    print("Confusion matrix shape:", confusion_mat.shape)
                
                specificity = tn / (tn + fp + 1e-7)
                fold_specificity += specificity

        avg_test_loss = fold_test_loss / len(test_loader)
        avg_recall = fold_recall / len(test_loader)
        avg_specificity = fold_specificity / len(test_loader)
        avg_f1 = f1_score(all_labels, all_preds, average='weighted')

        # 如果当前epoch的F1分数比之前的最佳F1分数高，则更新最佳epoch和最佳F1分数
        if avg_f1 > best_f1:
            best_epoch = epoch
            best_f1 = avg_f1

        print(f"Average_Loss: {avg_test_loss}")
        print(f"Average_Recall: {avg_recall}")
        print(f"Average_Specificity: {avg_specificity}")
        print(f"Average_F1: {avg_f1}")
        # 在训练循环中,将每个batch的loss添加到列表中
        test_losses.append(loss.item())

    # 将当前fold的最佳epoch和其对应的F1分数添加到列表中
    best_epochs.append(best_epoch)
    best_f1s.append(best_f1)

    # 更新总体指标
    total_test_loss += fold_test_loss
    total_recall += fold_recall
    total_specificity += fold_specificity
    total_f1 += avg_f1

# 找到F1分数最高的fold和其对应的epoch
best_fold = np.argmax(best_f1s)
best_epoch = best_epochs[best_fold]

        

print("整体Average_F1: {}".format(total_f1 / n_folds))
print("整体Average_Loss: {}".format(total_test_loss / (n_folds * len(test_loader))))
print("整体Average_Recall: {}".format(total_recall / (n_folds * len(test_loader))))
print("整体Average_Specificity: {}".format(total_specificity / (n_folds * len(test_loader))))
print(f"整体最好的epoch在第{best_fold + 1}次交叉验证,Epoch {best_epoch},F1分数为{best_f1s[best_fold]}")
print(len(losses))
print(len(test_losses))
plt.figure()
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss') 
plt.title('Training Loss')
# plt.show()
plt.savefig('/home/sunyvbo/demo1/dataset/CODE/figs/trainlosses.png')

plt.figure()
plt.plot(test_losses)
plt.xlabel('Iteration')
plt.ylabel('Loss') 
plt.title('Test Loss')
# plt.show()
plt.savefig('/home/sunyvbo/demo1/dataset/CODE/figs/testlosses.png')

