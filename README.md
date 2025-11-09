# 从零实现 Transformer (Transformer from Scratch)

本项目是BJTU《大模型基础与应用》课程的期中作业，旨在从零开始、仅使用 PyTorch 基础模块来完整实现一个标准的 Transformer 模型。模型架构严格遵循了 Vaswani 等人在其经典论文 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 中提出的设计。

我们实现了一个完整的编码器-解码器（Encoder-Decoder）架构，并在 IWSLT 2017 数据集的英德（En-De）翻译任务上进行了训练和评估。

## 主要实现功能

- **核心模块**: 手动实现了 Scaled Dot-Product Attention, Multi-Head Attention, Position-wise Feed-Forward Networks, 以及 Sinusoidal Positional Encoding。
- **完整架构**: 将核心模块组装成 `EncoderBlock` 和 `DecoderBlock`，并进一步堆叠成一个完整的 `Encoder`, `Decoder`, 以及最终的 `Transformer` 模型。
- **数据处理流水线**: 使用 `datasets` 和 `spacy` 库搭建了高效、可复现的数据预处理流程，包括分词、词典构建和动态填充。
- **训练与评估**: 实现了完整的训练、验证和评估脚本，支持 TensorBoard 监控、早停（Early Stopping）、梯度裁剪和 BLEU 分数评估。
- **可复现性**: 通过配置文件（`.yaml`）和固定的随机种子来保证所有实验的完全可复现性。

## 项目结构

```
.
|-- configs/                                        # 存放所有实验的 .yaml 配置文件
|   `-- 1-ffn_2048.yaml
|   `-- 2-ffn_512.yaml
|   `-- 3-mask_the_positionCode.yaml
|-- datasets                                        # 存放数据集en-de
|   |-- en-de/
|   |   |-- 略
|-- log/                                            # 存放训练日志文件
|   |-- 略
|-- models/                                         # 存放训练好的模型权重 (.pt)
|   `-- iwslt_en_de_base_ffn_512_best.pt
|   `-- iwslt_en_de_base_ffn_2048_best.pt
|   `-- iwslt_en_de_base_no_position_coding_best.pt
|-- results/                                        # 存放训练曲线图 (.png)
|   `-- iwslt_en_de_base_ffn_512_loss_curve.png
|   `-- iwslt_en_de_base_ffn_2048_loss_curve.png
|   `-- iwslt_en_de_base_no_position_coding_loss_curve.png
|-- scripts/                                        # 自动化训练脚本
|   `-- run_all_experiment.sh
|-- src/                                            # 项目核心源代码
|   |-- model/
|   |   |-- __init__.py
|   |   |-- attention.py
|   |   |-- decoder.py
|   |   |-- encoder.py
|   |   |-- modules.py
|   |   `-- transformer.py
|   |-- __init__.py
|   |-- config.py                                   # 加载配置文件的工具
|   |-- data.py                                     # 数据加载与预处理流水线
|   `-- utils.py                                    # 辅助函数 (掩码创建, 随机种子等)
|-- train.py                                        # 主训练脚本
|-- evaluate.py                                     # 评估脚本
|-- environment_backup.yml                          # Conda 环境备份文件
|-- requirements.txt                                # Pip 依赖文件
`-- README.md                                       # 项目说明
```

## 环境设置

我们强烈建议使用 Conda 来管理 Python 环境，以保证所有依赖（特别是 PyTorch 和 CUDA）的二进制兼容性。

### 1. 克隆代码仓库

```bash
git clone https://github.com/ShikiNatsume520/transformer-from-scratch.git
cd transformer-from-scratch
```

### 2. 创建 Conda 环境

我们提供了两种方式来创建环境，**强烈推荐使用第一种**。

#### 方式 A: 从 `yml` 文件一键创建 (推荐)

此方法可以完美复现我们实验时使用的所有包及其精确版本。

```bash
conda env create -f environment_backup.yml
conda activate deep_learning_py38
```

#### 方式 B: 手动安装依赖

如果你无法使用 `.yml` 文件，可以按照以下步骤手动创建。

```bash
# 1. 创建一个新的 Python 3.8 环境
conda create -n transformer_env python=3.8
conda activate transformer_env

# 2. 安装 PyTorch (请务必根据你的 CUDA 版本从官网获取命令)
# 例如，对于 CUDA 11.8:
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 torchtext==0.18.1 --index-url https://download.pytorch.org/whl/cu118

# 3. 安装其余依赖
pip install -r requirements.txt

# 4. 下载 Spacy 语言模型
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

## 复现实验

我们所有的实验都通过 `train.py` 和 `evaluate.py` 脚本执行，并通过 `--config` 参数指定对应的 `.yaml` 配置文件。
由于数据集是自动连接站点进行下载的，请在实验开始前确保你的网络连接状况稳定。

### 实时监控

在开始训练后，你可以在另一个终端中启动 TensorBoard 来实时监控损失曲线：
```bash
tensorboard --logdir=runs
```
### 一键训练命令
这条命令会自动读取configs目录下的所有yaml文件，并逐个运行train脚本
```bash
bash .\Scripts\run_all_experiment.sh      
```

### 实验 1: 基线模型 (Full Data, d_ff=2048) - 训练失败案例

此实验用于展示在没有学习率调度器的情况下，模型在完整数据集上训练的脆弱性。

**训练命令:**
```bash
python train.py --config configs/1-ffn_2048.yaml
```
**评估命令:**
```bash
python evaluate.py --config configs/1-ffn_2048.yaml --model models/iwslt_en_de_base_ffn_2048_best.pt
```

### 实验 2: 对照组 (20% Data, d_ff=512, with Positional Encoding)

此实验作为消融实验的对照组，在 20% 的数据子集上进行训练。

**训练命令:**
```bash
python train.py --config configs/2-ffn_512.yaml
```
**评估命令:**
```bash
python evaluate.py --config configs/2-ffn_512.yaml --model models/iwslt_en_de_base_ffn_512_best.pt
```

### 实验 3: 消融实验 (20% Data, d_ff=512, without Positional Encoding)

此实验移除了位置编码，以验证其作用。

**训练命令:**
```bash
python train.py --config configs/3-mask_the_positionCode.yaml
```
**评估命令:**
```bash
python evaluate.py --config configs/3-mask_the_positionCode.yaml --model models/iwslt_en_de_base_no_position_coding_best.pt
```

## 硬件要求

- **操作系统**: 在 Windows 11 上测试通过。
- **GPU**: 强烈推荐使用支持 CUDA 的 NVIDIA GPU。所有实验均在 **[NVIDIA GeForce RTX 4060 Laptop GPU 8GB]** 上完成。至少需要 8GB 显存才能以 `batch_size=16` 运行。
- **内存 (RAM)**: 数据预处理（特别是 `datasets.map()`）会占用较多内存。建议至少有 16 GB 内存，32 GB 更佳。
- **CPU**: 多核心 CPU（>= 8 核心）可以显著加速数据预处理。

## 许可证

本项目采用 [MIT License](LICENSE)。