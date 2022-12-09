import torch
from torch import nn

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, inputs, mask=None):
        # inputs:  batch_size, seq_len, emb_dim
        # mask: batch_size, seq_len

        if mask is not None:
            inputs = inputs * mask.unsqueeze(-1)

        return torch.sum(inputs, dim=1) / (torch.sum(mask, dim=1, keepdim=True) + 1e-6)
