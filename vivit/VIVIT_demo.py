"""
## Imports
"""

import cv2
import os
import io
import imageio
import ipywidgets
import numpy as np
import csv
import tensorflow as tf
from tensorflow import keras
from keras import layers

# 设置随机种子以确保可重现性
SEED = 42
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
keras.utils.set_random_seed(SEED)

# Hyperparameters
DATASET_NAME = "organmnist3d"
BATCH_SIZE = 32
VIDEO_SIZE = 16
AUTO = tf.data.AUTOTUNE
INPUT_SHAPE = (28, 28, 28, 1)
NUM_CLASSES = 2  # 修改为二类分类

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

EPOCHS = 60

PATCH_SIZE = (8, 8, 8)
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2

LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 8

# 数据处理代码...
# 从CSV文件中读取视频地址和对应视频标签
def read_labels_from_csv(file_path):
    video_paths = []
    labels = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            video_path, label = row
            video_paths.append(video_path)
            label = int(label)  #标签是整数形式
            labels.append(label)
    return video_paths, labels

# 加载视频数据集
def load_video_dataset(video_paths, labels, image_size):
    def generator():
        for video_path, label in zip(video_paths, labels):
            frames = []
            cap = cv2.VideoCapture(video_path)
            while True:         #src.get    src.read
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, image_size)
                frames.append(frame)
            cap.release()
            frames = tf.convert_to_tensor(frames, dtype=tf.float32)

            tf.print("Video frames shape:", tf.shape(frames))
            yield frames, label
            
    dataset = tf.data.Dataset.from_generator(generator, output_signature=(
        tf.TensorSpec(shape=(*image_size, 3), dtype=tf.float32),
        # tf.TensorSpec(shape=(), dtype=tf.int32)
    ))
    return dataset

# 从CSV文件读取训练数据集的视频地址和标签
train_video_paths, train_labels = read_labels_from_csv('/home/yanjiawen/ProjectTwo/dataset/video/train/label.csv')

# 加载训练数据集
train_dataset = load_video_dataset(train_video_paths, train_labels, image_size=(224, 224))
  
# 根据视频地址对训练数据集进行重新排序
train_video_paths = tf.convert_to_tensor(train_video_paths, dtype=tf.string)
train_video_indices = tf.range(len(train_video_paths))
train_video_indices = tf.random.shuffle(train_video_indices)
train_labels = tf.gather(train_labels, train_video_indices)
train_dataset = tf.data.Dataset.from_tensor_slices((train_video_paths, train_video_indices))
train_dataset = train_dataset.map(lambda video_path, index: (video_path, train_labels[index]))
 
# 设置训练数据集的批量大小
# train_dataset = train_dataset.batch(BATCH_SIZE)

# 从CSV文件读取验证数据集的视频地址和标签
valid_video_paths, valid_labels = read_labels_from_csv('/home/yanjiawen/ProjectTwo/dataset/video/valid/label.csv')

# 加载验证数据集
valid_dataset = load_video_dataset(valid_video_paths, valid_labels, image_size=(224, 224))

# 根据视频地址对验证数据集进行重新排序
valid_video_paths = tf.convert_to_tensor(valid_video_paths, dtype=tf.string)
valid_video_indices = tf.range(len(valid_video_paths))
valid_video_indices = tf.random.shuffle(valid_video_indices)
valid_labels = tf.gather(valid_labels, valid_video_indices)
valid_dataset = tf.data.Dataset.from_tensor_slices((valid_video_paths, valid_video_indices))
valid_dataset = valid_dataset.map(lambda video_path, index: (video_path, valid_labels[index]))

# 设置验证数据集的批量大小
# valid_dataset = valid_dataset.batch(BATCH_SIZE)

# 从CSV文件读取测试数据集的视频地址和标签
test_video_paths, test_labels = read_labels_from_csv('/home/yanjiawen/ProjectTwo/dataset/video/test/label.csv')

# 加载测试数据集
test_dataset = load_video_dataset(test_video_paths, test_labels, image_size=(224, 224))

# 根据视频地址对测试数据集进行重新排序
test_video_paths = tf.convert_to_tensor(test_video_paths, dtype=tf.string)
test_video_indices = tf.range(len(test_video_paths))
test_video_indices = tf.random.shuffle(test_video_indices)
test_labels = tf.gather(test_labels, test_video_indices)
test_dataset = tf.data.Dataset.from_tensor_slices((test_video_paths, test_video_indices))
test_dataset = test_dataset.map(lambda video_path, index: (video_path, test_labels[index]))

# 设置测试数据集的批量大小
# test_dataset = test_dataset.batch(BATCH_SIZE)

# 模型创建和训练代码...

class TubeletEmbedding(layers.Layer):
    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv3D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="VALID",
        )
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, videos):
        projected_patches = self.projection(videos)
        flattened_patches = self.flatten(projected_patches)
        return flattened_patches


class PositionalEncoder(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=self.embed_dim
        )
        self.positions = tf.range(start=0, limit=num_tokens, delta=1)

    def call(self, encoded_tokens):
        # Encode the positions and add it to the encoded tokens
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens

# 创建二类分类器的ViViT模型
def create_vivit_classifier(
    tubelet_embedder,
    positional_encoder,
    input_shape=INPUT_SHAPE,
    transformer_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    embed_dim=PROJECTION_DIM,
    layer_norm_eps=LAYER_NORM_EPS,
    num_classes=NUM_CLASSES,
):
    # Get the input layer
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    patches = tubelet_embedder(inputs)
    # Encode patches.
    encoded_patches = positional_encoder(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization and MHSA
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1
        )(x1, x1)

        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer Normalization and MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = keras.Sequential(
            [
                layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu),
                layers.Dense(units=embed_dim, activation=tf.nn.gelu),
            ]
        )(x3)

        # Skip connection
        encoded_patches = layers.Add()([x3, x2])

    # Layer normalization and Global average pooling.
    representation = layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)
    representation = layers.GlobalAvgPool1D()(representation)


    # Classify outputs.
    outputs = layers.Dense(units=num_classes, activation="sigmoid")(representation)  # 修改为sigmoid激活函数

    # 创建Keras模型
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# # 训练和评估代码...
def run_experiment():
    # 初始化模型
    model = create_vivit_classifier(
        tubelet_embedder=TubeletEmbedding(
            embed_dim=PROJECTION_DIM, patch_size=PATCH_SIZE
        ),
        positional_encoder=PositionalEncoder(embed_dim=PROJECTION_DIM),
    )

    # 编译模型，设置优化器、损失函数和评估指标
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",  # 使用二元交叉熵损失函数
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),  # 使用二元准确率作为评估指标
            keras.metrics.AUC(name="auc"),  # 使用AUC作为评估指标
        ],
    )

    # 训练模型
    model.fit(train_dataset, epochs=EPOCHS, validation_data=valid_dataset)

    # 评估模型
    _, accuracy = model.evaluate(test_dataset)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return model


model = run_experiment()
