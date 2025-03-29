import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        
        nn.init.kaiming_normal_(self.double_conv[0].weight)
        nn.init.kaiming_normal_(self.double_conv[3].weight)

    def forward(self, x):
        return self.double_conv(x)

class NoiseToVoidUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512, 1024]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downsampling
        in_features = in_channels
        for feature in features:
            self.downs.append(DoubleConv(in_features, feature))
            in_features = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Upsampling
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Final convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], out_channels, kernel_size=1),
            nn.Sigmoid()  # Sigmoid for binary segmentation
        )

        # Feature extraction layers
        self.feature_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(f, f//2, kernel_size=1),
                nn.BatchNorm2d(f//2),
                nn.LeakyReLU(inplace=True)
            ) for f in features
        ])

    def forward(self, x):
        skip_connections = []
        extracted_features = []

        # Downsampling and feature extraction
        for down, feature_layer in zip(self.downs, self.feature_layers):
            x = down(x)
            skip_connections.append(x)
            extracted_features.append(feature_layer(x))
            x = self.pool(x)

        x = self.bottleneck(x)

        # Reverse skip connections for upsampling
        skip_connections = skip_connections[::-1]

        # Upsampling
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                #x = torch.nn.functional.resize(x, size=skip_connection.shape[2:])
                x = torch.nn.functional.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=False)


            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x), extracted_features
    
    def __str__(self):
        return "N2NUNet"

def get_n2n_unet_model(in_channels=1, out_channels=1, device='cpu'):
    model = NoiseToVoidUNet(in_channels=in_channels, 
                           out_channels=out_channels,
                           features=[64, 128, 256, 512])
    return model.to(device)