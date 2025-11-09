"""
这里实现 Multi_head self-attention模块，以及辅助函数
"""
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    """计算缩放点积注意力"""
    d_k = q.size(-1)  # 获取最后一个维度的大小，即 d_k
    # 1. Q @ K^T / sqrt(d_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # 2. 应用掩码 (mask)
    if mask is not None:
        # 将 mask 中为 0 的位置填充一个非常小的数，使其在 softmax 后接近 0
        # scores = scores.masked_fill(mask == 0, float('-inf'))
        scores = scores.masked_fill(mask == 0, -1e9)

    # 3. Softmax
    p_attn = F.softmax(scores, dim=-1)

    # 4. (可选) Dropout
    if dropout is not None:
        p_attn = dropout(p_attn)

    # 5. @ V
    return torch.matmul(p_attn, v), p_attn


class MultiHeadAttention(nn.Module):
    """
        实现 multi-head self-attention.
        Input = [Q, K, V]
        Output = Softmax(Mask(Q*V/d_k)) * V
        """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"   # 只有整除，才能将输入分成多个头

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度

        # 定义四个独立的线性层
        self.W_q = nn.Linear(d_model, d_model)  # 投影层
        self.W_k = nn.Linear(d_model, d_model)  # 投影层
        self.W_v = nn.Linear(d_model, d_model)  # 投影层
        self.W_o = nn.Linear(d_model, d_model)  # 将所有头的结果拼接后，做最终的线性变换。

        # Dropout 层作用于注意力分数
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1. 线性投影: (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        #    这里 self.W_q(query) 等价于 query @ self.W_q.weight.t() + self.W_q.bias
        # 原论文中描述的是投影到不同空间，重复n_head次，将每次得到的结果进行拼接后维度等于d_model。 而实现时，我们使用一个等维度投影向量，
        # 然后再直接等距分割，这两个效果是一致的，但是第二种方法避免了for循环，从而利用GPU加快了速度
        q = self.W_q(query)
        k = self.W_k(key)
        v = self.W_v(value)

        # 2. 拆分成 n_heads 个头:
        #    (batch, seq_len, d_model) -> (batch, seq_len, n_heads, d_k)
        #    然后 -> (batch, n_heads, seq_len, d_k) 以便并行计算
        # 通过transpose， 将n_heads维度提前，这样送入乘法函数时，batch,n_heads都会被看做是批次
        q = q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 3. 计算注意力: 对所有头并行计算
        #    x 的形状是 (batch, n_heads, seq_len_q, d_k)
        #    self.attn (注意力权重) 的形状是 (batch, n_heads, seq_len_q, seq_len_k)
        assert mask.dim() == 4, "multi_head must receive the mask that it's dim==4"
        x, attn = scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.dropout)

        # 4. 拼接多头的结果:
        #    (batch, n_heads, seq_len_q, d_k) -> (batch, seq_len_q, n_heads, d_k)
        #    然后 -> (batch, seq_len_q, d_model)
        # 执行像 .transpose() 这样的操作时，PyTorch 并不会真的移动内存中的数据，因为那样做成本太高。相反，它只是修改了张量的元数据
        # .view()，对张量的内存布局有严格要求：它只能作用于连续的张量上
        # .contiguous() 是在 transpose 这种“虚拟”操作和 view 这种需要“物理”基础的操作之间，一个必不可少的“物理化”步骤
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 5. 最后的线性投影
        return self.W_o(x)