from turtle import forward
import math
import torch
import torch.nn as nn


class LinearHead(nn.Module):
    def __init__(self, inplanes, outplanes, dropout):
        super().__init__()
        self.linear = nn.Linear(inplanes, outplanes)
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
    
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        self.linear.weight.data.uniform_(-stdv, stdv)
        if self.linear.bias is not None:
            self.linear.bias.data.zero_()
    
    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        return x
    

