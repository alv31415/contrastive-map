import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Downsampler(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsampler, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.mpconv(x)


class Upsampler(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsampler, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_1, x_2 = None):
        x_1 = self.up(x_1)
        x = x_1 if x_2 is None else torch.cat([x_2, x_1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, contrastive_model=None):
        super(UNet, self).__init__()
        self.contrastive_model = contrastive_model
        self.RESNET_DIM = 224

        self.kwargs = {"contrastive_model": contrastive_model}
        self.use_contrastive_model = False

        # freeze contrastive model, we don't want to train it, just encode images
        if contrastive_model is not None:
            self.use_contrastive_model = True
            self.contrastive_img_width = 8
            self.contrastive_hidden_dim = 512
            self.contrastive_bottleneck = nn.Linear(in_features = self.contrastive_hidden_dim,
                                                    out_features = self.contrastive_img_width
                                                                   * self.contrastive_img_width
                                                                   * self.contrastive_hidden_dim)

            for param in contrastive_model.parameters():
                param.requires_grad = False

        self.input = None if self.use_contrastive_model else DoubleConv(in_channels = 3, out_channels = 64)
        self.downsampler1 = None if self.use_contrastive_model else Downsampler(in_channels = 64, out_channels = 128)
        self.downsampler2 = None if self.use_contrastive_model else Downsampler(in_channels = 128, out_channels = 256)
        self.downsampler3 = None if self.use_contrastive_model else Downsampler(in_channels = 256, out_channels = 512)
        self.downsampler4 = None if self.use_contrastive_model else Downsampler(in_channels = 512, out_channels = 512)

        self.upsampler1 = Upsampler(in_channels = 512, out_channels = 256)
        self.upsampler2 = Upsampler(in_channels = 256, out_channels = 128)
        self.upsampler3 = Upsampler(in_channels = 128, out_channels = 64)
        self.upsampler4 = Upsampler(in_channels = 64, out_channels = 32)
        self.output = nn.Sequential(nn.Conv2d(in_channels = 32, out_channels = 3, kernel_size = 1),
                                    nn.Tanh())

    def forward(self, x):
        if self.use_contrastive_model:
            x_down = self.contrastive_bottleneck(x)
            x_down = x_down.view(-1,
                                 self.contrastive_hidden_dim,
                                 self.contrastive_img_width,
                                 self.contrastive_img_width)
            x_up = self.upsampler1(x_down, None)
            x_up = self.upsampler2(x_up, None)
            x_up = self.upsampler3(x_up, None)
            x_up = self.upsampler4(x_up, None)
            x_up = self.output(x_up)
        else:
            x_down_1 = self.input(x)
            x_down_2 = self.down1(x_down_1)
            x_down_3 = self.down2(x_down_2)
            x_down_4 = self.down3(x_down_3)
            x_down_5 = self.down4(x_down_4)
            x_up = self.up1(x_down_5, x_down_4)
            x_up = self.up2(x_up, x_down_3)
            x_up = self.up3(x_up, x_down_2)
            x_up = self.up4(x_up, x_down_1)
            x_up = self.output(x_up)

        return x_up








