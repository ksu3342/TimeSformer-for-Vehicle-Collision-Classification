import csv
import cv2
import numpy as np
import tensorflow as tf
import tqdm
import random


# 指定每个视频要读取的帧数
frames_num= 100

#指定步长
frame_step = 15

def frames_from_video_file(video_path, n_frames, output_size=(224, 224), frame_step=15):
  # 内部函数，用于对视频帧进行格式化
  def format_frames(frame, output_size):
    frame = tf.image.convert_image_dtype(frame, tf.float32)  # 将图像转换为浮点数类型
    frame = tf.image.resize_with_pad(frame, *output_size)    # 调整图像大小并填充
    return frame

  result = []                                 # 用于存储最终的帧数据
  src = cv2.VideoCapture(str(video_path))     # 创建视频捕获对象 src

  video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)   # 获取视频的总帧数

  need_length = 1 + (n_frames - 1) * frame_step   # 计算需要提取的帧的总长度

  if need_length > video_length:
    start = 0
  else:
    max_start = video_length - need_length
    start = random.randint(0, max_start + 1)

  src.set(cv2.CAP_PROP_POS_FRAMES, start)   # 设置视频读取位置为随机的起始位置
  ret, frame = src.read()   # 读取第一帧
  result.append(format_frames(frame, output_size))   # 对第一帧进行格式化并添加到结果列表

  for _ in range(n_frames - 1):
    for _ in range(frame_step):
      ret, frame = src.read()   # 按照帧步长读取视频帧
    if ret:
      frame = format_frames(frame, output_size)   # 对读取的帧进行格式化
      result.append(frame)   # 将格式化后的帧添加到结果列表
    else:
      result.append(np.zeros_like(result[0]))   # 如果读取失败，将空白帧添加到结果列表
  src.release()   # 释放视频捕获对象
  result = np.array(result)[..., [2, 1, 0]]   # 转换结果列表为 NumPy 数组，并调整 BGR 通道为 RGB 通道

  return result   # 返回格式化后的帧数据数组

print(frames_from_video_file('/home/yanjiawen/ProjectTwo/dataset/TEXT/0/000830.mp4', frames_num, output_size=(224, 224), frame_step=15))