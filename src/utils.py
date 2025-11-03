# In: src/utils.py
import torch
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
    将训练和验证损失曲线保存为图片。
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

    save_path = os.path.join(results_dir, f"{experiment_name}_loss_curve.png")
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