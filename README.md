# **TimeSformer-Collision-Detection**  
**基于时空Transformer的汽车碰撞实时检测系统**  
[![GitHub license](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![Python 3.8](https://img.shields.io/badge/Python-3.8+-green)](https://www.python.org/)  
[![PyTorch 1.12](https://img.shields.io/badge/PyTorch-1.12+-blue)](https://pytorch.org/)  

---

## **项目概述**  
针对传统汽车碰撞检测模型复杂度高、实时性不足的问题，本项目基于 **TimeSformer** 架构提出一种轻量化、高精度的时空特征提取方案，实现复杂交通场景下的实时碰撞预警。核心成果包括：  
- **F1分数**：0.976（4折交叉验证均值）  
- **推理延迟**：85ms（Jetson Nano，CUDA加速，INT8量化）  
- **特异性提升**：通过`divided_space_time`注意力机制，较全局注意力提升4.7%  



## **技术亮点**  
### **1. 时空特征优化**  
- **帧数消融实验**：  
  - 时序帧长从64调整至100帧，F1分数提升3.2%（0.944→0.976）。  
- **注意力机制改进**：  
  - `divided_space_time`注意力机制：  
    - 特异性提升4.7%（90.5%→95.2%）；  
    - 通过分离时空维度计算，减少冗余计算量。  
- **类别不平衡处理**：  
  - 加权采样（碰撞样本权重5:1）+ 渐进式学习率策略（余弦退火）；  
  - 真阳性率提升11.3%（88.4%→99.7%）。  

### **2. 模型轻量化与实时性**  
- **参数压缩**：  
  - 移除冗余Transformer头（12→8头）+ 通道剪枝；  
  - 参数量减少20%（25M→20M），推理速度提升43%。  
- **边缘部署优化**：  
  - 在Jetson Nano实现**85ms/帧**延迟（Batch=1，INT8量化）；  
  - 对比：YOLO V3（209ms）与MobileNet V2（57ms）精度更高。  

---

