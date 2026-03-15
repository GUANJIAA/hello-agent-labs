from turtle import forward, position
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    位置编码层，用于将序列中的每个位置映射到一个固定维度的向量表示。
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建一个足够长的位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # pe(positional encoding) 的大小为 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # 对所有偶数索引位置，使用sin函数编码
        pe[:, 0::2] = torch.sin(position * div_term)
        # 对所有奇数索引位置，使用cos函数编码
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 形状为 (1, max_len, d_model)
        # 将pe 注册为缓冲区，这样它就不会被视为模型的可训练参数
        self.register_buffer('pe', pe)

    def forward(self, x) -> torch.Tensor:
        # x.size(1) 表示序列的长度
        # 对输入x 加上对应的位置编码
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    """
    多头注意力层，用于在序列中进行注意力计算。
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能够被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 定义Q, K, V 线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        # 1.计算注意力得分
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 2.应用编码（如果提供）
        if mask is not None:
            # 将掩码中为0的位置的注意力得分设置为负无穷大，这样softmax后会接近0
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 3.计算注意力权重
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # 4.加权求和（权重*V）
        attn_output = torch.matmul(attn_weights, value)
        return attn_output

    def spilt_heads(self, x):
        # 将输入x的形状从(batch_size, seq_len, d_model)
        # 转换为(batch_size, num_heads, seq_len, d_k)
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # 将输入x的形状从(batch_size, num_heads, seq_len, d_k)
        # 转换为(batch_size, seq_len, d_model)
        batch_size, num_heads, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, query, key, value, mask):
        # 1. 对Q, K, V 进行线性变换
        query = self.spilt_heads(self.W_q(query))
        key = self.spilt_heads(self.W_k(key))
        value = self.spilt_heads(self.W_v(value))

        # 2.计算缩放点积注意力
        attn_output = self.scaled_dot_product_attention(query, key, value, mask)

        # 3. 合并多头输出并进行最终的线性变换
        output = self.W_o(self.combine_heads(attn_output))
        return output

class PositionWiseFeedForward(nn.Module):
    """
    位置前馈网络层，用于在每个位置上进行前馈计算。
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        # 1. 线性变换
        x = self.linear1(x)
        # 2. ReLU激活
        x = self.relu(x)
        # 3. Dropout
        x = self.dropout(x)
        # 4. 线性变换
        x = self.linear2(x)
        # output: (batch_size, seq_len, d_model)
        return x

# 编码器核心层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention() # 待实现
        self.feed_forward = PositionWiseFeedForward() # 待实现
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 残差连接与层归一化
        # 1.多头自注意力
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 2.前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# --- 解码核心层 ---
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention() # 待实现
        self.cross_attn = MultiHeadAttention() # 待实现
        self.feed_forward = PositionWiseFeedForward() # 待实现
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # 残差连接与层归一化
        # 1.自注意力
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 2.交叉注意力
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        # 3.前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x