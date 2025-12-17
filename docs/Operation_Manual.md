# 实验三操作手册 (Operation Manual)

本手册详细指导如何运行实验三的各个部分，包括代码结构说明、运行步骤和结果验证。

---

## 第一部分：基于 MNIST 数据集的自编码器 (MNIST Autoencoder)

### 1.1 实验目标
- 实现深度自编码器，压缩 MNIST 图片至 2 维潜空间。
- 实现降噪自编码器，去除高斯白噪声。
- 可视化重构效果、潜空间分布以及潜空间采样生成。

### 1.2 运行方式

#### 方法 A：使用 Jupyter Notebook (推荐)
这是最直观的方式，适合理解原理和逐步调试。
1.  进入 `MNIST_Autoencoder` 文件夹。
2.  打开 `MNIST_Autoencoder.ipynb`。
3.  点击 "Run All" 或逐步运行单元格。
    - **Cell 1-3**: 加载数据并定义模型。
    - **Cell 4**: 训练普通自编码器 (Standard AE)，观察 Loss 下降曲线。
    - **Cell 5**: 训练降噪自编码器 (Denoising AE)，观察 Loss。
    - **Cell 6**: 展示数字 0-9 的 **原始图 vs 加噪图 vs 重构图**。
    - **Cell 7**: 展示 2D 潜空间的聚类散点图 (不同数字颜色不同)。
    - **Cell 8**: 在潜空间均匀采样，生成连续变化的数字矩阵。

#### 方法 B：使用命令行脚本
如果你喜欢使用终端，也可以运行 Python 脚本。
1.  打开终端，进入 `MNIST_Autoencoder` 目录：
    ```bash
    cd MNIST_Autoencoder
    ```
2.  **训练模型**：
    ```bash
    python train.py                 # 训练普通 AE，保存为 results/simple_ae.pth
    python train.py --denoising     # 训练降噪 AE，保存为 results/denoising_ae.pth
    ```
3.  **可视化结果**：
    ```bash
    python visualize.py
    ```
    运行后，请查看 `results/` 文件夹下的图片：
    - `simple_reconstruction.png`
    - `denoising_reconstruction.png`
    - `latent_space.png`
    - `latent_sampling.png`

---

## 第二部分：图像风格迁移 (Image Style Transfer)

### 2.1 实验目标
- 复现 AdaIN (Adaptive Instance Normalization) 模型。
- 训练模型并使用 Tensorboard 可视化。
- 完成自定义任务：北邮景点迁移和 Alpha 混合。

### 2.2 准备工作
1.  进入 `Style_Transfer` 目录：
    ```bash
    cd Style_Transfer
    ```
2.  克隆 AdaIN 代码库：
    ```bash
    git clone https://github.com/naoto0804/pytorch-AdaIN.git
    ```
3.  **数据准备**：
    - 本实验需要 MS-COCO (内容图) 和 WikiArt (风格图) 数据集。
    - 如果你没有这些数据集，无法进行完整的训练过程。建议使用预训练模型进行后续的推断任务。
    - 将预训练模型 (`decoder.pth`, `vgg_normalised.pth`) 放入 `pytorch-AdaIN/models/` 目录中。

### 2.3 训练模型 (如果具备数据)
```bash
python pytorch-AdaIN/train.py --content_dir <COCO路径> --style_dir <WikiArt路径>
```
*要求：训练至少 10,000 次迭代，并使用 Tensorboard 记录 Loss。*

### 2.4 自定义任务 (Custom Tasks)
我们提供了一个脚本 `custom_tasks.py` 来简化这些任务。

#### 任务 3：北邮景点风格迁移
1.  准备两张你在北邮拍摄的照片 (例如 `bupt1.jpg`, `bupt2.jpg`)。
2.  准备一张风格图片 (例如 `style.jpg`)。
3.  运行命令：
    ```bash
    # 确保你在 Style_Transfer 目录下
    python custom_tasks.py --content bupt1.jpg --style style.jpg --output result_bupt1.png
    python custom_tasks.py --content bupt2.jpg --style style.jpg --output result_bupt2.png
    ```

#### 任务 4：Alpha 混合 (基于学号)
该任务要求根据学号尾数使用不同的 Alpha 值 (内容与风格的融合比例)。
- **偶数尾号**: Alpha = [0.3, 0.6, 0.9]
- **奇数尾号**: Alpha = [0.2, 0.5, 0.8]

运行命令 (替换 `YOUR_ID` 为你的学号)：
```bash
python custom_tasks.py --content bupt1.jpg --style style.jpg --student_id 2021211123 --output alpha_test.png
```
脚本会自动识别你的学号尾数，并生成三张不同 Alpha 值的图片 (例如 `alpha_test_alpha_0.2.png` 等)。

---

## 常见问题 (FAQ)

**Q: 运行 `visualize.py` 报错 `FileNotFoundError`?**
A: 请确保你先运行了 `train.py` 生成了模型文件 (`.pth`)。

**Q: 风格迁移脚本提示找不到 `net` 模块?**
A: 请确保你已经克隆了 `pytorch-AdaIN` 仓库，并且是在 `Style_Transfer` 目录下运行 `custom_tasks.py`。脚本会尝试将 `pytorch-AdaIN` 加入 Python 路径。

**Q: 如何查看 Tensorboard?**
A: 在训练目录下运行 `tensorboard --logdir ./logs`，然后在浏览器访问 `localhost:6006`。
