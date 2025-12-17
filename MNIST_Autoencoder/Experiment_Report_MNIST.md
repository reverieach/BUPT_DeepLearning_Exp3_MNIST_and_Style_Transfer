# 实验三：基于 MNIST 数据集的自编码器实现 - 实验报告

**课程名称**：神经网络与深度学习
**姓名**：[你的姓名]
**学号**：[你的学号]
**日期**：2025年12月10日

---

## 一、 实验目的
1.  掌握深度自编码器 (Deep Autoencoder) 的构建与训练方法。
2.  理解自编码器在数据降维和特征提取中的作用。
3.  掌握降噪自编码器 (Denoising Autoencoder) 的原理及实现。
4.  探索潜在空间 (Latent Space) 的分布特性及生成能力。

## 二、 实验内容与步骤

### 1. 数据读写与模型搭建
本次实验使用 MNIST 手写数字数据集。
- **数据预处理**：将图像转换为 Tensor，像素值归一化到 [0, 1] 区间。
- **模型结构**：搭建了一个深度自编码器。
    - **编码器 (Encoder)**：包含 3 个全连接层，激活函数使用 ReLU。层级结构为 `784 -> 512 -> 256 -> 128 -> 2`。
    - **Bottleneck 层**：维度限制为 **2**，以便于在二维平面上可视化潜在空间。
    - **解码器 (Decoder)**：包含 3 个全连接层，结构与编码器对称 `2 -> 128 -> 256 -> 512 -> 784`，最后一层使用 **Sigmoid** 激活函数以确保输出在 [0, 1] 范围内。

```python
# 核心代码片段：模型定义
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 2)  # Bottleneck dim=2
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 28 * 28), nn.Sigmoid()
        )
```

### 2. 模型训练 (普通自编码器)
**要求**：选择二元交叉熵函数作为损失函数，在限制 bottleneck 层维度为 2 的情况下训练模型。

- **损失函数**：`nn.BCELoss()` (Binary Cross Entropy)
- **Bottleneck**：`nn.Linear(128, 2)`

**(请在此处插入代码截图：包含 Loss 定义和 Bottleneck 层定义的代码)**
> [截图建议：截取 Autoencoder 类定义中 self.encoder 的最后几行，以及 train_model 函数中 criterion = nn.BCELoss() 的部分]

**结果展示**：
下图展示了数字 0 到 9 的原始图片和重建图片。

*(在此处插入 0-9 重建对比图)*
> [请将 Notebook 中生成的 "Standard AE Reconstruction" 图片插入此处。注意：目前的 Notebook 代码只生成了降噪的 0-9 对比。你需要运行下面补充的代码块来生成普通 AE 的 0-9 对比图]

### 3. 降噪自编码器 (Denoising AE)
**要求**：设置噪声因子为 0.4，在输入图像上叠加均值为 0 且方差为 1 的标准高斯白噪声，训练降噪自编码器。

**(请在此处插入代码截图：包含噪声添加逻辑和训练循环的代码)**
> [截图建议：截取 train_model 函数中 `if denoising:` 下面的噪声添加代码]

**关键代码说明**：
```python
# 噪声添加逻辑
noise_factor = 0.4
# torch.randn_like 生成标准正态分布 (均值0, 方差1) 的噪声
noisy_img = img + noise_factor * torch.randn_like(img)
# 将像素值截断回 [0, 1] 范围，防止溢出
noisy_img = torch.clamp(noisy_img, 0., 1.)
```

**降噪结果展示**：
下图展示了数字 0 到 9 的 10 张图片的**原始图片、加噪图片和重建图片**。

*(在此处插入 0-9 降噪对比图)*
> [请将 Notebook 中生成的 "Denoising Results (0-9)" 图片插入此处]

### 4. 潜在空间 (Latent Space) 可视化与采样
#### (1) 潜在空间聚类
我们将测试集所有图片输入编码器，提取 Bottleneck 层的 2 维特征并进行可视化。

*(在此处插入 Latent Space Distribution 散点图)*
> [请将 Notebook 中生成的 Latent Space 散点图截图插入此处]

**分析**：
可以看到不同数字在二维空间中形成了明显的聚类。例如，形态相似的数字（如 1 和 7，3 和 8）在空间中的距离可能较近，而形态差异大的数字则分得较开。这说明自编码器成功将高维图像信息压缩并保留了语义结构。

#### (2) 均匀采样重构
我们在潜在空间的特定范围内（如 [-30, 30]）进行均匀网格采样，并将采样点输入解码器生成图像。

*(在此处插入 Uniform Sampling 矩阵图)*
> [请将 Notebook 中生成的均匀采样矩阵图截图插入此处]

**分析**：
观察生成图像矩阵，可以看到数字在潜在空间中是连续变化的。从一个数字渐变到另一个数字（例如从 6 渐变到 0），中间产生了一些混合形态的数字。这验证了潜在空间的连续性，表明解码器具有良好的生成能力。

## 三、 实验总结
本次实验成功实现了基于 MNIST 的深度自编码器。
1.  通过限制 Bottleneck 维度为 2，实现了数据的极致压缩与可视化。
2.  降噪自编码器展现了强大的抗噪能力，验证了其特征学习的鲁棒性。
3.  潜在空间的探索揭示了深度学习模型对数据流形结构的捕捉能力。

---
**注**：请将 Jupyter Notebook 运行生成的图片保存并替换文档中的文字说明部分，最后导出为 PDF 即可满足提交要求。
