import torch
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.models.hub import slowfast_r50
from torch import nn

model = slowfast_r50(pretrained=True)
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale
from torchvision.transforms import Compose
from torchvision.transforms import CenterCrop

transform = ApplyTransformToKey(
  key="video",
  transform=Compose([
    ShortSideScale(256),
    CenterCrop(256)
  ])
)
# 模型改为2分类

model.head = nn.Linear(2304, 2)

# 视频处理transform
transform = ApplyTransformToKey(
    key="video",
    transform=Compose([
        ShortSideScale(256),
        CenterCrop(256)
    ])
)

# 加载正样本视频
video_path = '/home/lihaoyang/code/training/positive/02148-前.mp4'
video = EncodedVideo.from_path(video_path)
video_data = video.get_clip(start_sec=0, end_sec=1)
inputs = transform(video_data)['video']

# 模型预测
with torch.no_grad():
    preds = model(inputs)
    pred_class = preds.argmax(dim=1)

if pred_class == 0:
    print('Negative')
else:
    print('Positive')

# 同样方式加载负样本预测