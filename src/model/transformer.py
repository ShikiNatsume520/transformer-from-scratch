# 在这里，实现transformer模型
import torch
import torch.nn as nn
from src.model.encoder import Encoder
from src.model.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self,
                 source_vocab_size: int,
                 target_vocab_size: int,
                 d_model: int,
                 n_layers: int, # 假定encoder与decoder堆叠的块的数量相同
                 n_heads: int,
                 d_ff: int,
                 dropout: float,
                 max_len: int = 5000):
        super().__init__()

        # --- 实例化 Encoder 和 Decoder ---
        # Encoder 负责处理源序列
        self.encoder = Encoder(source_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len)

        # Decoder 负责处理目标序列并与 Encoder 的输出交互
        self.decoder = Decoder(target_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len)

        # --- 最后的输出层 ---
        # 将 Decoder 的输出从 d_model 维度映射到目标词汇表大小
        self.final_linear = nn.Linear(in_features=d_model, out_features=target_vocab_size)    # 输出下一个词的概率

    def forward(self, source_seq: torch.Tensor, target_seq: torch.Tensor, source_mask: torch.Tensor, target_mask: torch.Tensor):
        """
        定义 Transformer 的一次完整前向传播 (主要用于训练)。

        Args:
            source_seq (torch.Tensor): 源序列的 token IDs, 形状 (batch_size, source_seq_len)
            target_seq (torch.Tensor): 目标序列的 token IDs, 形状 (batch_size, target_seq_len)
            source_mask (torch.Tensor): 源序列的掩码
            target_mask (torch.Tensor): 目标序列的掩码

        Returns:
            torch.Tensor: 模型的原始输出 (logits), 形状 (batch_size, target_seq_len, target_vocab_size)
        """
        # 1. 将源序列和其掩码传入 Encoder
        # encoder_output 包含了源句子的全部上下文信息
        # 形状: (batch_size, source_seq_len, d_model)
        encoder_ouput = self.encoder(source_seq, source_mask)

        # 2. 将目标序列、Encoder 的输出以及两个掩码传入 Decoder
        # decoder_output 是目标序列在给定源序列上下文下的表示
        # 形状: (batch_size, target_seq_len, d_model)
        decoder_output = self.decoder(target_seq, encoder_ouput, source_mask, target_mask)

        # 3. 将 Decoder 的输出传入最终的线性层，得到 logits
        # 形状: (batch_size, target_seq_len, target_vocab_size)
        logits = self.final_linear(decoder_output)

        return logits
