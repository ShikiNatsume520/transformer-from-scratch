"""
这里实现decoder块
"""
import torch
import torch.nn as nn
import math
from src.model.modules import PositionwiseFeedForward, PositionalEncoding
from src.model.attention import MultiHeadAttention

class DecoderBlock(nn.Module):
    """
    实现 Transformer 解码器中的一个块 (Decoder Block)。
    它包含三个子层：掩码自注意力、交叉注意力、前馈网络。
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        """
        初始化解码器块的各个组件。
        """
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)  # 第一个子层: 掩码多头自注意力
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout) # 第二个子层: 多头交叉注意力

        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)  # 第三个子层: 逐位前馈网络

        # --- Add & Norm 组件 ---
        # 对应三个子层的层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # 对应三个子层的 Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, source_mask: torch.Tensor, target_mask: torch.Tensor):
        """
        定义 Decoder Block 的前向传播逻辑。

        Args:
            x (torch.Tensor): (batch_size, target_seq_len, d_model) 来自解码器前一层的输入。
            encoder_output (torch.Tensor): (batch_size, source_seq_len, d_model) 整个编码器栈的最终输出。
            source_mask (torch.Tensor): 源序列的掩码，用于交叉注意力。
            target_mask (torch.Tensor): 目标序列的掩码，用于掩码自注意力。

        Returns:
            torch.Tensor: (batch_size, target_seq_len, d_model) 解码器块的输出张量。
        """
        # --- 1. 第一个子层: 掩码多头自注意力 ---
        # Q, K, V 全部来自解码器自身的输入 x。
        # 使用 target_mask 来防止当前位置注意到未来的位置。
        residual = x    # 保留残差值
        attention_output_1 = self.self_attn(x, x, x, target_mask)  # MHA
        x = self.norm1(residual + self.dropout1(attention_output_1))    # Add & Norm

        # --- 2. 第二个子层: 交叉注意力 ---
        # Query 来自前一个子层的输出 x。
        # Key 和 Value 来自编码器的最终输出 encoder_output。
        # 使用 source_mask 来忽略源序列中的填充部分。
        residual = x
        attention_output_2 = self.cross_attn(x, encoder_output, encoder_output, source_mask) # MHA
        x = self.norm2(residual + self.dropout2(attention_output_2))    # Add & Norm

        # --- 3. 第三个子层: 前馈网络 ---
        residual = x
        ffn_output = self.ffn(x)    # FFN
        x = self.norm3(residual + self.dropout3(ffn_output))    # Add & Norm

        return x


class Decoder(nn.Module):
    """
    完整的 Decoder 结构，由 N 个 DecoderBlock 堆叠而成。
    """
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, d_ff: int, dropout: float, max_len: int = 5000):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)  # 词嵌入层
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)    # 嵌入位置编码

        # 使用 nn.ModuleList 存储 N 个独立的 DecoderBlock
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

        # 最后的层归一化
        self.norm = nn.LayerNorm(d_model)


    def forward(self, target_seq: torch.Tensor, encoder_output: torch.Tensor, source_mask: torch.Tensor, target_mask: torch.Tensor):
        """
        Decoder 的前向传播。

        Args:
            target_seq (torch.Tensor): 目标序列的 token IDs, 形状 (batch_size, target_seq_len)
            encoder_output (torch.Tensor): Encoder 的最终输出, 形状 (batch_size, source_seq_len, d_model)
            source_mask (torch.Tensor): 源序列掩码
            target_mask (torch.Tensor): 目标序列掩码

        Returns:
            torch.Tensor: Decoder 的输出, 形状 (batch_size, target_seq_len, d_model)
        """
        # 1. 词嵌入和位置编码
        # 注意：同样，论文中 embedding 的权重会乘以 sqrt(d_model)
        embedding_output = self.embedding(target_seq) * math.sqrt(self.embedding.embedding_dim)
        x = self.positional_encoding(embedding_output)

        # 2. 依次通过 N 个 DecoderBlock
        for layer in self.layers:
            x = layer(x, encoder_output, source_mask, target_mask)

        # 3. 最后的层归一化
        return self.norm(x)
