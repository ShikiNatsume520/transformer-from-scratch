"""
这里实现train函数
"""
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter # 导入 TensorBoard
import os

from src.model.transformer import Transformer
from src.config import load_config
from src.last_data import get_dataloaders
from src.utils import set_seed, save_plot, create_masks


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, pad_idx, writer, epoch):
    """
    训练一个 epoch。
    """
    model.train()  # 将模型设置为训练模式
    total_loss = 0

    # 使用 tqdm 创建进度条
    for i, (source_input, target_input, target_label) in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch + 1}")):
        # print(f"source_batch: {source_batch}")
        # print(f"target_batch: {target_batch}")
        source_input = source_input.to(device)
        target_input = target_input.to(device)
        target_label = target_label.to(device)

        # 3. 创建掩码
        source_mask, target_mask = create_masks(source_input, target_input, pad_idx)

        # 4. 前向传播
        predictions = model(source_input, target_input, source_mask, target_mask)

        # 5. 计算损失
        loss = loss_fn(predictions.view(-1, predictions.size(-1)), target_label.contiguous().view(-1))

        # 6. 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        # --- TensorBoard 记录 ---
        # 每 100 个 batch 记录一次瞬时损失
        if i % 100 == 0:
            writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(dataloader) + i)

    avg_loss = total_loss / len(dataloader)     # 计算平均损失
    writer.add_scalar('Loss/train_epoch', avg_loss, epoch)  # TensorBoard 记录损失

    return avg_loss

def validate(model, dataloader, loss_fn, device, pad_idx):
    """
    在一个 epoch 结束后进行验证。
    """
    model.eval()  # 将模型设置为评估模式
    total_loss = 0
    with torch.no_grad():  # 在此模式下，不计算梯度，节省计算资源
        for source_input, target_input, target_label in tqdm(dataloader, desc="Validation"):
            # 1. 将数据移动到指定设备 (GPU/CPU)
            source_input = source_input.to(device)
            target_input = target_input.to(device)
            target_label = target_label.to(device)

            # 3. 创建掩码
            source_mask, target_mask = create_masks(source_input, target_input, pad_idx)

            # 4. 前向传播
            predictions = model(source_input, target_input, source_mask, target_mask)

            # 5. 计算损失
            loss = loss_fn(predictions.view(-1, predictions.size(-1)), target_label.contiguous().view(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)

def main(config):
    # 设置随机种子
    set_seed(config['seed'])

    # 设置 TensorBoard
    # 结果会保存在 runs/iwslt_en_de_base 这样的文件夹里
    writer = SummaryWriter(log_dir=os.path.join("runs", config['experiment_name']))

    # 1. 指定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 加载数据集
    train_loader, val_loader, en_vocab, de_vocab, max_src_len, max_tgt_len = get_dataloaders(config)
    print("Data loaders created.")

    # 从词典中获取 vocab_size 和 pad_idx
    config['source_vocab_size'] = len(en_vocab)
    config['target_vocab_size'] = len(de_vocab)
    # pad_idx = de_vocab.stoi['<pad>']  # 示例 padding 字符对应的嵌入值
    source_pad_idx = en_vocab['<pad>']
    target_pad_idx = de_vocab['<pad>']

    # 取源和目标中的最大值，并增加一个小的 buffer
    config['max_len'] = max(max_src_len, max_tgt_len) + 10
    print(f"Model max_len set to: {config['max_len']}")

    # 3. 初始化模型
    model = Transformer(
        source_vocab_size=config['source_vocab_size'],
        target_vocab_size=config['target_vocab_size'],
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        max_len=config['max_len'],
        use_positional_encoding=config.get('use_positional_encoding', True)
    ).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # 4. 定义损失函数和优化器
    # ignore_index=pad_idx 告诉损失函数忽略所有 padding token, 即我们无需计算被填充位置上的词的熵，因为这里本来就没有词
    loss_fn = nn.CrossEntropyLoss(ignore_index=target_pad_idx)
    if config['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.98), eps=1e-9)
    else:  # 默认 Adam
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.98), eps=1e-9)

    # 5. 训练循环
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    # --- 早停逻辑所需的变量 ---
    patience_counter = 0
    early_stopping_patience = config.get('early_stopping_patience', 5)  # 从配置中读取，提供默认值
    min_delta = config.get('early_stopping_min_delta', 0.001)

    for epoch in range(config['epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, target_pad_idx, writer, epoch)
        val_loss = validate(model, val_loader, loss_fn, device, target_pad_idx)

        # --- 记录和保存 ---
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        writer.add_scalar('Loss/validation_epoch', val_loss, epoch)

        print(f"Epoch {epoch + 1}/{config['epochs']}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # --- 早停和模型保存逻辑 ---
        # 检查是否有显著改进
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0  # 重置耐心计数器

            # 保存最佳模型
            os.makedirs("models", exist_ok=True)
            model_save_path = f"models/{config['experiment_name']}_best.pt"
            torch.save(model.state_dict(), model_save_path)
            print(f"Validation loss improved. Model saved to {model_save_path}")
        else:
            patience_counter += 1  # 没有改进，耐心计数器 +1
            print(f"No improvement in validation loss for {patience_counter} epoch(s).")

        # 判断是否触发早停
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {patience_counter} epochs without improvement.")
            break  # 中断训练循环

    print("Training finished.")

    # 6. 保存最终图表
    save_plot(train_losses, val_losses, config['experiment_name'])
    writer.close()


if __name__ == "__main__":
    # 1. 设置参数解析器
    parser = argparse.ArgumentParser(description="Train a Transformer model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    # 2. 加载配置
    config = load_config(args.config)

    # 3. 运行主训练函数
    main(config)