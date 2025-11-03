"""
本脚本用于实现一些可复用的子模块：
1. position-wise FFN
2. PositionalEncoding
"""
import math
import torch
import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    """
    实现 Position-wise Feed-Forward Network.
    net_structure = {
        Linear(in_features=d_model, out_features=d_ff)
        ReLU()
        Dropout()
        Linear(in_features=d_ff, out_features=d_model)
    }
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model (int): FFN的输入输出维度，由于Transformer其实不会改变维度，所以也是整个模型的输入输出.
            d_ff (int): FFN隐藏层的维度.
            dropout (float): 丢弃率.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=d_ff, out_features=d_model)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量, 形状为 (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: 输出张量, 形状为 (batch_size, seq_len, d_model)
        """
        return self.net(x)


class PositionalEncoding(nn.Module):
    """
    实现位置编码功能
    """
    def __init__(self, d_model, dropout, max_len=5000):
        """

        :param d_model:  输入输出的特征维度
        :param dropout:  丢弃率
        :param max_len:  最大句子长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x 的形状是 (batch_size, seq_len, d_model)
        # self.pe 的形状是 (1, max_len, d_model)
        # 我们需要截取与 x 序列长度相同的部分
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)    # 为词添加位置编码（简单相加即可）
        return self.dropout(x)
