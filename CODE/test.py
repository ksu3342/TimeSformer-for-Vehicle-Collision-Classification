# 导入所需的包
import torch  
from torch import nn
from torch.utils.data import DataLoader

from train import VideoDataset, VideoTransformer # 导入定义的Dataset和模型

# 构建模型
model = VideoTransformer(embed_dim=256, num_heads=8, num_layers=3, num_classes=2)
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
model = model.to(device) 

# 加载预训练模型参数  
model.load_state_dict(torch.load('model_weights.pth', map_location=device))

# 设置模型为评估模式
model.eval()   

# 构建测试集
test_dataset = VideoDataset('/home/sunyvbo/TEXT') 
test_loader = DataLoader(test_dataset, batch_size=16)

# 预测统计
correct = 0  
total = 0

# 关闭梯度计算
with torch.no_grad():
    for i, (videos, labels) in enumerate(test_loader):
        # 数据送入设备
        videos = videos.to(device)  
        labels = labels.to(device)
        
        # 前向传播,得到预测结果
        outputs = model(videos) 
        
        # 获得预测类别
        _, predicted = torch.max(outputs.data, 1)
        
        # 统计预测正确的样本个数
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# 计算准确率 
accuracy = correct / total
print(f'Test Accuracy: {accuracy * 100}%')
# import torch
# from torch import nn
# from torch.utils.data import DataLoader
#
# from loader import VideoDataset, VideoTransformer
#
# # Instantiate the model
# model = VideoTransformer(embed_dim=256, num_heads=8, num_layers=3, num_classes=2)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
#
# # Load the saved weights
# model.load_state_dict(torch.load('model_weights.pth', map_location=device))
#
# # Don't forget to set the model to evaluation mode
# model.eval()
#
# # Assume you have a test set prepared
# test_dataset = VideoDataset('testing')
# test_loader = DataLoader(test_dataset, batch_size=2)
#
# correct = 0
# total = 0
#
# # No gradient calculation is needed
# with torch.no_grad():
#     for i, (videos, labels) in enumerate(test_loader):
#         videos = videos.to(device)
#         labels = labels.to(device)
#
#         # Perform forward pass
#         outputs = model(videos)
#
#         # Compute predictions
#         _, predicted = torch.max(outputs.data, 1)
#
#         # Calculate accuracy
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# accuracy = correct / total
# print(f'Test Accuracy: {accuracy * 100}%')
