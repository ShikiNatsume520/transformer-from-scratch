# In: src/utils.py
import torch
import torch.nn as nn
import random
import numpy as np
import os
import matplotlib.pyplot as plt


def set_seed(seed: int):
    """
    固定随机种子以确保实验的可复现性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 确保 cudnn 的确定性，可能会牺牲一些性能
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_plot(train_losses, val_losses, experiment_name: str, results_dir: str = "results"):
    """
    将训练和验证损失曲线保存为图片，自动避免重名，尾部增加序号。
    """
    # 确保结果目录存在
    os.makedirs(results_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Training and Validation Loss for {experiment_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 构建初始保存路径
    base_filename = f"{experiment_name}_loss_curve"
    save_path = os.path.join(results_dir, f"{base_filename}.png")

    # 如果文件已存在，则在文件名末尾增加序号
    counter = 1
    while os.path.exists(save_path):
        save_path = os.path.join(results_dir, f"{base_filename}_{counter}.png")
        counter += 1

    # 保存图片
    plt.savefig(save_path)
    print(f"Loss curve saved to {save_path}")
    plt.close()

def create_masks(source_batch, target_batch, pad_idx):
    # 1. 创建源序列的填充掩码
    # 形状: (batch_size, 1, 1, source_seq_len)
    source_mask = (source_batch != pad_idx).unsqueeze(1).unsqueeze(2)

    # 2. 创建目标序列的前瞻掩码和填充掩码的组合
    target_seq_len = target_batch.size(1)

    # 目标序列填充掩码
    # 形状: (batch_size, 1, 1, target_seq_len)
    target_padding_mask = (target_batch != pad_idx).unsqueeze(1).unsqueeze(2)

    # 前瞻掩码 (Look-ahead mask)
    # 形状: (1, 1, target_seq_len, target_seq_len)
    lookahead_mask = torch.tril(torch.ones((target_seq_len, target_seq_len), device=target_batch.device)).bool()

    # 最终的目标掩码是两者的结合
    # 形状: (batch_size, 1, target_seq_len, target_seq_len)
    target_mask = target_padding_mask & lookahead_mask

    return source_mask, target_mask


# src/utils.py (或者你定义 create_masks 的地方)

import torch


def create_masks_for_torch_transformer(source_batch, target_batch, pad_idx, device):
    """
    为 torch.nn.Transformer 创建掩码。
    """
    # 1. 源填充掩码 (src_key_padding_mask)
    #    形状: (batch_size, source_seq_len)
    #    含义: True 表示该位置是 padding，需要被忽略。
    src_key_padding_mask = (source_batch == pad_idx)

    # 2. 目标因果掩码 (tgt_mask)
    #    形状: (target_seq_len, target_seq_len)
    #    含义: 上三角矩阵，值为 -inf 的位置表示不能关注。
    target_seq_len = target_batch.size(1)
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(target_seq_len, device=device)

    # 3. 目标填充掩码 (tgt_key_padding_mask)
    #    形状: (batch_size, target_seq_len)
    #    含义: 与 src_key_padding_mask 相同。
    tgt_key_padding_mask = (target_batch == pad_idx)

    # 4. Encoder 输出的填充掩码 (memory_key_padding_mask)
    #    这个掩码在 Decoder 的 Cross-Attention 中使用，用于忽略 Encoder 输出中的 padding。
    #    它和 src_key_padding_mask 是完全一样的。
    memory_key_padding_mask = src_key_padding_mask

    return src_key_padding_mask, tgt_key_padding_mask, tgt_mask, memory_key_padding_mask