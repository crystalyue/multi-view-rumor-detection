import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from config import opt


# Attention: K(L, B, H), V(L, B, H), Q(L, B, H) -> c(B, L, H), attn(B, L, L)
class Attention(nn.Module):  # output: tuple: (context, attn)
    def __init__(self, args):
        super(Attention, self).__init__()
        self.args = args

    def forward(self, X, X_padding_mask=None, coverage=None, dropout=0.1):
        """
        K / key: (L, B, H) encoder_outputs, encoder feature
        V / value: (L, B, H) to calculate the context vector
        Q / query: (L, B, H) last_hidden, deocder feature
        X_padding_mask: (B, 1, L)
        coverage: (B, L)
        """
        X_dim = X.size(-1)
        X_query = X.transpose(0, 1)  # -> (B, L, H)
        X_key = X.transpose(0, 1)  # -> (B, L, H)
        X_value = X.transpose(0, 1)  # -> (B, L, H)

        scores = torch.matmul(X_query, X_key.transpose(-2, -1)) / math.sqrt(X_dim)  # (B, L, H) x (B, H, L) -> (B, L, L)

        attn_dist = F.softmax(scores, dim=-1)  # (B, L, L)
        attn_dist = F.dropout(attn_dist, p=dropout)
        context = torch.matmul(attn_dist, X_value)  # (B, L, L) x (B, L, H) -> (B, L, H)

        # calculate average
        context = context.sum(1)/context.size(1)
        return context, attn_dist


