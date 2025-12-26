import torch
import torch.nn as nn
import math

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, kernel_size=3,stride=1,padding=0):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels  , kernel_size=kernel_size, padding=padding, bias=False),
            nn.InstanceNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels  , kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(out_channels  )
            )
    def forward(self, x):
        return (self.residual_function(x) + self.shortcut(x))


class AttentionForAdjacency(nn.Module):
    def __init__(self,embed_dims, dropout=0.1, **kwargs):
        "Take in model size and number of heads."
        super(AttentionForAdjacency, self).__init__()
        self.linearK = nn.Linear(embed_dims, embed_dims)
        self.linearQ = nn.Linear(embed_dims, embed_dims)
        self.normK = nn.LayerNorm(embed_dims)
        self.normQ = nn.LayerNorm(embed_dims)
        self.dropout = dropout
        self.conv_1 = nn.Conv2d(1, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(1, 1, kernel_size=1)
        self.basic_block = BasicBlock(1,1,kernel_size=1)
        

    def forward(self,x):
        Q, K = self.normQ(self.linearQ(x)), self.normK(self.linearK(x))
        adj_matrix = torch.matmul(Q,K.transpose(-2, -1))/math.sqrt(Q.shape[-1])
        adj_matrix = self.basic_block(adj_matrix.unsqueeze(1)).squeeze(1)
        return adj_matrix