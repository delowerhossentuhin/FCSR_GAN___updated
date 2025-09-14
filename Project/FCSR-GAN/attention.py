import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleSpatialChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.in_channels = in_channels
        # Channel attention
        self.ca_avg = nn.AdaptiveAvgPool2d(1)
        self.ca_max = nn.AdaptiveMaxPool2d(1)
        self.ca_fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.ca_sigmoid = nn.Sigmoid()
        # Spatial attention (multi-scale)
        self.sa_conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.sa_conv2 = nn.Conv2d(2, 1, kernel_size=5, padding=2)
        self.sa_conv3 = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sa_sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, h, w = x.size()
        # Channel attention
        avg_out = self.ca_avg(x)
        max_out = self.ca_max(x)
        ca_out = self.ca_fc(avg_out) + self.ca_fc(max_out)
        ca_out = self.ca_sigmoid(ca_out)
        x_ca = x * ca_out
        # Spatial attention
        avg_sa = torch.mean(x_ca, dim=1, keepdim=True)
        max_sa = torch.max(x_ca, dim=1, keepdim=True)[0]
        sa_in = torch.cat([avg_sa, max_sa], dim=1)
        sa1 = self.sa_sigmoid(self.sa_conv1(sa_in))
        sa2 = self.sa_sigmoid(self.sa_conv2(sa_in))
        sa3 = self.sa_sigmoid(self.sa_conv3(sa_in))
        sa_out = (sa1 + sa2 + sa3) / 3
        x_out = x_ca * sa_out
        return x_out