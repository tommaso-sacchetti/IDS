import math
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


class PositionalEncoder(nn.Module):
    """Positional encoder class"""
    
    # The positional encoding vector, embedding_dim is d_model
    def __init__(self, num_features, max_seq_length=256, dropout=0.1):
        # num_features = embedding_dim
        super(PositionalEncoder, self).__init__()
        self.num_features = num_features
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_length, num_features)
        for pos in range(max_seq_length):
            for i in range(0, num_features, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / num_features)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i + 1) / num_features)))
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x * math.sqrt(self.num_features)
        seq_length = x.size(1)
        pe = Variable(self.pe[:, :seq_length], requires_grad=False).to(x.device)
        # Add the positional encoding vector to the embedding vector
        x = x + pe
        x = self.dropout(x)
        return x


class SelfAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        # if mask needed in this method is where to put it
        # in this case input length is all the same so mask not needed
        key_dim = key.size(-1)
        attn = torch.matmul(query / np.sqrt(key_dim), key.transpose(2, 3))
        attn = self.dropout(torch.softmax(attn, dim=-1))
        output = torch.matmul(attn, value)

        return output


class Attention(nn.Module):
    """Attention layer (single head)"""

    def __init__(self, num_features, dropout=0.1):
        # num_features = embedding_dim
        self.num_features = num_features
        self.self_attention = SelfAttention(dropout)
        self.query_projection = nn.Linear(num_features, num_features)
        self.key_projection = nn.Linear(num_features, num_features)
        self.value_projection = nn.Linear(num_features, num_features)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(num_features, num_features)

    def forward(self, query, key, value):
        # Apply the linear projections
        batch_size = query.size(0)
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)
        # Reshape the input
        query = query.view(batch_size, -1, 1, self.num_features).transpose(1, 2)
        key = key.view(batch_size, -1, 1, self.num_features).transpose(1, 2)
        value = value.view(batch_size, -1, 1, self.num_features).transpose(1, 2)
        # Calculate the attention
        scores = self.self_attention(query, key, value)
        # Reshape the output
        output = (
            scores.transpose(1, 2).contiguous().view(batch_size, -1, self.num_features)
        )
        # Apply the linear projection
        output = self.out(output)
        return output


