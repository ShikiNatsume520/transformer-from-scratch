"""
这里定义编码块
"""
import torch.nn as nn
from modules import MultiHeadAttention, PositionwiseFeedForward

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




