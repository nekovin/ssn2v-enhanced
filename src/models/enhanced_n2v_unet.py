import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import functional as F
import os 
import sys
sys.path.append(r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\src")
from loss import Noise2VoidLoss


class ResDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        )
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        self.activation = nn.LeakyReLU(inplace=True)
        
        nn.init.kaiming_normal_(self.double_conv[0].weight)
        nn.init.kaiming_normal_(self.double_conv[3].weight)

    def forward(self, x):
        identity = self.residual(x)
        x = self.double_conv(x)
        return self.activation(x + identity)
    
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)
    
class EnhancedNoiseToVoidUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512, 1024]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.channel_attentions = nn.ModuleList()

        # Downsampling with Channel Attention
        in_features = in_channels
        for feature in features:
            down_block = ResDoubleConv(in_features, feature)
            self.downs.append(down_block)
            self.channel_attentions.append(ChannelAttention(feature))
            in_features = feature

        # Bottleneck
        self.bottleneck = ResDoubleConv(features[-1], features[-1] * 2)

        # Upsampling with advanced interpolation
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(ResDoubleConv(feature * 2, feature))

        # Final convolution with adaptive output
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], out_channels, kernel_size=1),
            nn.Tanh()  # More flexible than Sigmoid
        )

    def forward(self, x):
        skip_connections = []

        # Downsampling with channel attention
        for down, attention in zip(self.downs, self.channel_attentions):
            x = down(x)
            x = attention(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        # Upsampling
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = torch.nn.functional.interpolate(
                    x, 
                    size=skip_connection.shape[2:], 
                    mode="bilinear", 
                    align_corners=False
                )

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)
    
def get_e_n2n_unet_model(in_channels=1, out_channels=1, device='cpu'):
    model = EnhancedNoiseToVoidUNet(in_channels=in_channels, 
                           out_channels=out_channels,
                           features=[64, 128, 256, 512])
    return model.to(device)

def get_e_unet_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_e_n2n_unet_model()
    model = model.to(device)

    criterion = Noise2VoidLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    return device, model, criterion, optimizer
