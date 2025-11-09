"""
这里定义编码块
"""
import torch.nn as nn
import math
from src.model.modules import PositionwiseFeedForward, PositionalEncoding
from src.model.attention import MultiHeadAttention

class EncoderBlock(nn.Module):
    """
    实现一个基本的编码块，结构如下：
    输入 x
     |
     +----------------------------------------------------------+ // 残差连接 1
     |                                                          |
     |---->[ Sub-layer 1: Multi-Head Self-Attention ]----+      |
     |                                                   |      |
     +-----------------> Dropout -----------------> Add --+-> LayerNorm -> 输出 x'
                                                         |
     +---------------------------------------------------+-------+ // 残差连接 2
     |                                                           |
     |---->[ Sub-layer 2: Position-wise FFN ]------------+       |
     |                                                   |       |
     +-----------------> Dropout -----------------> Add --+-> LayerNorm -> 最终输出
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        """

        :param d_model: 输入输出的特征维度
        :param n_heads: 多头注意力机制的参数
        :param d_ff: FFN隐藏层参数
        :param dropout: 丢弃率
        """
        super().__init__()

        self.mult_head_attention = MultiHeadAttention(d_model, n_heads, dropout)    # 一个多头注意力块
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)      # 一个前馈网络块
        self.norm1 = nn.LayerNorm(d_model)  # 多头注意力后的Norm层
        self.norm2 = nn.LayerNorm(d_model)  # 前馈块后的Norm层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


    def forward(self, x, mask):
        """

        :param x: 输入的特征
        :param mask: 掩码，用于遮盖后续时间步
        :return:
        """
        residual = x    # 保存残差值
        attention_output  = self.mult_head_attention(x, x, x, mask) # MHA
        x = self.norm1(residual + self.dropout1(attention_output))  # Add & Norm

        residual = x    # 保存残差值
        ffn_output = self.ffn(x)    #FFN
        x = self.norm2(residual + self.dropout2(ffn_output))    # Add & Norm

        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, d_ff: int, dropout: float, max_len: int = 5000, use_positional_encoding: bool = True):
        """

        :param vocab_size:  词汇表大小
        :param d_model: 模型输入输出维度
        :param n_layers: block块的堆叠层数
        :param n_heads: 多头注意力机制的层数
        :param d_ff: FFN的隐藏层维度
        :param dropout: 丢弃率
        :param max_len: 一条句子的单词数量上限
        """
        super().__init__()

        self.use_pe = use_positional_encoding

        self.embedding = nn.Embedding(vocab_size, d_model)  # 词嵌入层

        if self.use_pe:
            self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)    # 为嵌入添加位置编码

        # 使用 nn.ModuleList 来存储 N 个独立的 EncoderBlock
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

        # 最终的层归一化
        self.norm = nn.LayerNorm(d_model)


    def forward(self, source_seq, mask):
        """
        Encoder 的前向传播。

        Args:
            source_seq (torch.Tensor): 源序列的 token IDs, 形状 (batch_size, source_seq_len)
            mask (torch.Tensor): 源序列的掩码

        Returns:
            torch.Tensor: Encoder 的输出, 形状 (batch_size, source_seq_len, d_model)
       """
        # 1. 词嵌入和位置编码
        # 注意：论文中 embedding 的权重会乘以 sqrt(d_model)
        embedding_output = self.embedding(source_seq) * math.sqrt(self.embedding.embedding_dim)

        # 2. (可选) 添加位置编码
        if self.use_pe:
            x = self.positional_encoding(embedding_output)
        else:
            x = embedding_output

        # 2. 依次通过 N 个 EncoderBlock
        for layer in self.layers:
            x = layer(x, mask)

        # 3. 最后的层归一化
        return self.norm(x)


