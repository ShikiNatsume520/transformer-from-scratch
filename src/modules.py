"""
本脚本用于实现一些可复用的子模块：
1. Multi_head self-attention
2. position-wise FFN
3. Res_block + Layer_normal
4. Position_encode
"""
import torch
import torch.nn as nn

class FFN(nn.Module):
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

