import torch
import torch.nn as nn

from .basemodels import *

class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 downsample=False):
        super(ResidualBlock, self).__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size,
                               stride,                           
                               padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1,
                               padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, 
                          out_channels, 1, 
                          stride, 
                          bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        res = self.downsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + res
        x = self.relu(x)
        return x

class ResnetCNN(BaseModel):
    def __init__(self,
                 in_channels=1,
                 num_filters=16,
                 num_filters_list=[],
                 output_dim=1,
                 kernel_size=3,
                 kernel_sizes=[],
                 num_conv_layers=3,
                 dropout=0.5):
        super(ResnetCNN, self).__init__()
        if not num_filters_list:
            num_filters_list = [num_filters] * num_conv_layers
            
        if not kernel_sizes:
            kernel_sizes = [kernel_size] * num_conv_layers

        if len(num_filters_list) != num_conv_layers:
            num_conv_layers = len(num_filters_list)
        layers = []
        for idx in range(num_conv_layers):
            conv_layer = ResidualBlock(in_channels=in_channels,
                                out_channels=num_filters_list[idx],
                                kernel_size=kernel_sizes[idx])
            layers.append(conv_layer)
            layers.append(nn.BatchNorm2d(num_filters_list[idx]))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = num_filters_list[idx]
        
        self.conv_layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveMaxPool2d(output_size=1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(num_filters_list[-1], num_filters_list[-1]) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(num_filters_list[-1], output_dim)
        
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x