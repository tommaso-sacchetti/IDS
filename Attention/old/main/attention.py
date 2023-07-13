import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def attention(query, key, value):
    # Equation 1 in Vaswani et al. (2017)
    # Scaled dot product between Query and Keys
    output = torch.matmul(query, key.transpose(-2, -1)) / (query.size(2) ** 0.5)

    # Softmax to get attention weights
    attention_weights = F.softmax(output, dim=-1)

    # Multiply weights by Values
    weighted_sum = torch.matmul(attention_weights, value)

    # Following Figurr 1 and Section 3.1 in Vaswani et al. (2017)
    # Residual connection i.e. add weighted sum to original query
    output = weighted_sum + query

    # Layer normalization
    layer_norm = nn.LayerNorm(output.size()[2:], eps=1e-5)
    output = layer_norm(output)

    return output, attention_weights


def pos_encoding(input_x, num_hidden, seq_len, keep_prob):
    # pass initial input through dense to become embedded
    # TODO: check if input_x.size(-1) is correct
    # TODO: no activation function in linear layer, might be better to add ReLu
    dense_layer = nn.Linear(input_x.size(-1), num_hidden)
    embedded_input = dense_layer(input_x)

    # Giving inputs positional information
    # positional_encoding = tf.Variable into nn.Parameter
    positional_encoding = nn.Parameter(
        torch.zeros(1, seq_len, num_hidden), requires_grad=True
    )

    positional_input = embedded_input + positional_encoding

    # Apply dropout
    dropout = nn.Dropout(p=1 - keep_prob)
    positional_input = dropout(positional_input)

    return positional_input


def sdsa(encoding):
    """SDSA module"""
    encoding, enc_attention_weights = attention(encoding, encoding, encoding)

    print("sdsa_out", encoding)

    return encoding, enc_attention_weights


def ffn(encoding, num_hidden):
    """FF module"""
    dense = nn.Linear(encoding.size(-1), num_hidden * 8)
    dense_output = dense(encoding)
    dense_output = nn.ReLU()(dense_output)

    print("ffn_out", dense_output)

    encoding = encoding + nn.Linear(dense_output.size(-1), num_hidden)(dense_output)

    print("ffn_add", encoding)

    layer_norm = nn.LayerNorm(encoding.size()[2:], eps=1e-5)
    encoding = layer_norm(encoding)

    return encoding


def sda(encoding, num_hidden, initial_input, seq_len):
    """SDA module"""
    decoder_input = nn.Parameter(torch.zeros(1, seq_len, num_hidden), requires_grad=True)
    print("sda_query", decoder_input)

    tiled_decoder_input = decoder_input.repeat(initial_input.size(0), 1, 1)

    decoded, decoder_attention_weights = attention(tiled_decoder_input, encoding, encoding)

    print("sda_out", decoded)
    print("sda_out_weights", decoder_attention_weights)

    return decoded, decoder_attention_weights
 

def apply_attention(model_output, initial_input, num_hidden, seq_len):
    """A fucntion which can be used for combination study"""

    ffn_out = ffn(model_output, num_hidden)
    sda_out, sda_attention_weights = sda(ffn_out, num_hidden, initial_input, seq_len)
    sdsa_out, sdsa_attention_weights = sdsa(sda_out)

    return sdsa_out, sdsa_attention_weights, sda_attention_weights

