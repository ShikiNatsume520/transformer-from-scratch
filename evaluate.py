import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from torchmetrics.text import BLEUScore
from typing import List # <-- 在这里添加这一行
import math

# --- 从 src 导入项目模块 ---
from src.model.transformer import Transformer
from src.config import load_config
# 我们需要复用数据处理的逻辑
from src.last_data import (
    load_raw_data,
    get_tokenizers,
    build_vocabs_and_get_stats,
    TranslationDataset,
    PAD_IDX, SOS_IDX, EOS_IDX  # 确保这些特殊索引在 data.py 中是全局可访问的
)
from src.utils import create_masks


def greedy_decode(model: Transformer, source_seq: torch.Tensor, source_mask: torch.Tensor, max_len: int,
                  device: torch.device) -> List[int]:
    """
    一个简单的贪心解码函数，用于在推理时生成翻译。

    Args:
        model (Transformer): 训练好的 Transformer 模型。
        source_seq (torch.Tensor): 源序列的 token ID 张量，形状 (1, source_seq_len)。
        source_mask (torch.Tensor): 源序列的填充掩码。
        max_len (int): 生成句子的最大长度。
        device (torch.device): 计算设备。

    Returns:
        list[int]: 生成的目标序列的 token ID 列表。
    """
    model.eval()

    # 1. 将源序列输入 Encoder，得到 encoder_output (只计算一次)
    with torch.no_grad():
        encoder_output = model.encoder(source_seq, source_mask)

    # 2. 初始化 Decoder 的输入，只包含一个 <s> (SOS) 标记
    decoder_input = torch.tensor([[SOS_IDX]], dtype=torch.long, device=device)

    # 3. 自回归地生成目标序列的其余部分
    for _ in range(max_len - 1):  # -1 是因为已经有 SOS 了
        # 创建目标掩码 (此时 target_mask 只是一个 look-ahead mask，因为没有 padding)
        # 注意：source_mask 仍然需要传入 decoder
        _, target_mask = create_masks(source_seq, decoder_input, PAD_IDX)

        # Decoder 前向传播
        with torch.no_grad():
            output = model.decoder(decoder_input, encoder_output, source_mask, target_mask)

        # 4. 获取最后一个时间步的输出，并通过最终线性层得到 logits
        prob = model.final_linear(output[:, -1])

        # 5. 选择概率最高的词作为下一个词
        _, next_word = torch.max(prob, dim=1)
        next_word_id = next_word.item()

        # 6. 将新生成的词拼接到 Decoder 输入上，用于下一轮预测
        decoder_input = torch.cat(
            [decoder_input, torch.tensor([[next_word_id]], dtype=torch.long, device=device)],
            dim=1
        )

        # 7. 如果生成了 </s> (EOS) 标记，则解码结束
        if next_word_id == EOS_IDX:
            break

    return decoder_input.squeeze(0).tolist()


def evaluate(config_path: str, model_path: str):
    """
    加载模型权重，在测试集上进行评估，并计算 BLEU 分数。
    """
    # --- 1. 加载配置和设置环境 ---
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. 加载数据和词典 (完全仿照 train.py 的逻辑) ---
    # a. 加载原始数据和分词器
    raw_datasets = load_raw_data(config['dataset_name'], config['language_pair'])
    en_tokenizer, de_tokenizer = get_tokenizers()

    # b. 关键修正：调用与训练时完全相同的函数来构建词典
    #    这将确保词典是在完整的 'train' split 上构建的。
    print("[INFO] Rebuilding vocabularies from the training set to ensure consistency...")
    en_vocab, de_vocab, max_src_len, max_tgt_len = build_vocabs_and_get_stats(
        raw_datasets, en_tokenizer, de_tokenizer
    )
    print("Vocabularies built successfully.")

    test_split = raw_datasets['test']
    test_subset_ratio = 0.1  # 默认为 1.0 (使用全部数据)

    if test_subset_ratio < 1.0:
        print(f"\n[INFO] 使用测试集 {test_subset_ratio * 100:.0f}% 的数据进行评估...")
        num_samples = math.ceil(len(test_split) * test_subset_ratio)
        test_split = test_split.select(range(num_samples))
        print(f"[INFO] 测试子集大小: {len(test_split)} 条样本。")

    # c. 为【测试集】创建 Dataset 和 DataLoader
    test_dataset = TranslationDataset(
        test_split, en_vocab, de_vocab, en_tokenizer, de_tokenizer
    )
    # 评估时 batch_size=1，shuffle=False
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print("Test data loaded.")

    # --- 3. 初始化模型并加载训练好的权重 ---
    # 关键：使用刚刚构建的词典大小来初始化模型
    config['source_vocab_size'] = len(en_vocab)  # en 是源
    config['target_vocab_size'] = len(de_vocab)  # de 是目标
    config['max_len'] = max(max_src_len, max_tgt_len) + 10

    model = Transformer(
        source_vocab_size=config['source_vocab_size'],
        target_vocab_size=config['target_vocab_size'],
        d_model=config['d_model'], n_layers=config['n_layers'], n_heads=config['n_heads'],
        d_ff=config['d_ff'], dropout=config['dropout'], max_len=config['max_len'],
        use_positional_encoding=config.get('use_positional_encoding', True)
    ).to(device)

    print(f"Loading model weights from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- 4. 初始化评估指标 ---
    bleu_metric = BLEUScore(n_gram=4)

    # --- 5. 遍历测试集进行推理和评估 ---
    predictions = []
    references = []

    # 注意：我们自己的 Vocab 类有 .ids_to_text 方法
    for source_seq, target_seq in tqdm(test_loader, desc="Evaluating on Test Set"):
        source_seq = source_seq.to(device)

        source_mask = (source_seq != PAD_IDX).unsqueeze(1).unsqueeze(2)

        predicted_ids = greedy_decode(model, source_seq, source_mask, max_len=config['max_len'], device=device)

        # 使用我们自己定义的 .ids_to_text 方法
        predicted_text = de_vocab.ids_to_text([idx for idx in predicted_ids if idx not in (SOS_IDX, EOS_IDX, PAD_IDX)])
        reference_text = de_vocab.ids_to_text(
            [idx for idx in target_seq.squeeze(0).tolist() if idx not in (SOS_IDX, EOS_IDX, PAD_IDX)])

        predictions.append(predicted_text)
        references.append([reference_text])

        # --- 6. 计算并打印最终的 BLEU 分数 ---
    bleu_score = bleu_metric(predictions, references)
    print(f"\n" + "=" * 50)
    print(f"Evaluation Finished!")
    print(f"BLEU Score on the test set: {bleu_score.item() * 100:.2f}")
    print("=" * 50)

    # --- 7. 打印一些翻译样本以进行直观感受 ---
    print("\n--- Sample Translations ---")
    for i in range(min(5, len(predictions))):
        source_ids_to_convert = [idx for idx in test_loader.dataset[i][0].tolist() if
                                 idx not in (SOS_IDX, EOS_IDX, PAD_IDX)]
        # 调用英语词典的 .ids_to_text() 方法
        source_text = en_vocab.ids_to_text(source_ids_to_convert)

        print(f"\n--- Sample {i + 1} ---")
        print(f"  Source:      {source_text}")
        print(f"  Reference:   {references[i][0]}")
        print(f"  Prediction:  {predictions[i]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Transformer model.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the .yaml configuration file used for training.")
    parser.add_argument("--model", type=str, required=True, help="Path to the saved model weights (.pt) file.")
    args = parser.parse_args()

    evaluate(args.config, args.model)