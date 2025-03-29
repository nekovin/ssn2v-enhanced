import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Double convolution block with batch normalization"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Ensure x1 and x2 have the same dimensions
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Final convolution"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class N2VUNet(nn.Module):
    """U-Net architecture for Noise2Void"""
    def __init__(self, n_channels=1, n_classes=1, base_filters=32):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Initial convolution
        self.inc = DoubleConv(n_channels, base_filters)
        
        # Downsampling path (encoder)
        self.down1 = Down(base_filters, base_filters*2)
        self.down2 = Down(base_filters*2, base_filters*4)
        
        # Bottleneck
        self.bottleneck = DoubleConv(base_filters*4, base_filters*8)
        
        # Upsampling path (decoder)
        self.up1 = Up(base_filters*8, base_filters*4)
        self.up2 = Up(base_filters*4, base_filters*2)
        self.up3 = Up(base_filters*2, base_filters)
        
        # Final convolution
        self.outc = OutConv(base_filters, n_classes)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        # Bottleneck
        x4 = self.bottleneck(x3)
        
        # Decoder path with skip connections
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        # Final output
        return self.outc(x)