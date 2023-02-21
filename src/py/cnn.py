import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

logging.basicConfig(format="%(asctime)s %(levelname)-4s %(message)s",
                    level=logging.INFO,
                    datefmt="%d-%m-%Y %H:%M:%S")

import torch.nn as nn


class CNNCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_batch_norm=True, use_pool=True):
        super(CNNCell, self).__init__()

        self.cell = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels) if use_batch_norm else nn.Identity(),
            nn.MaxPool2d(kernel_size=2, stride=2) if use_pool else nn.Identity())

    def forward(self, x):
        return self.cell(x)


class GlobalAveragePooling(nn.Module):
    def __init__(self, dim):
        super(GlobalAveragePooling, self).__init__()
        self.dim = dim

        self.avg_pooler = nn.AvgPool2d(self.dim)

    def forward(self, x):
        avg = self.avg_pooler(x)

        return avg.squeeze()


class CNN(nn.Module):
    def __init__(self, input_dim, in_channels, output_dim, use_bias=True, use_batch_norm=True, use_flatten=True):
        super(CNN, self).__init__()

        self.cnn_cells = 5
        self.global_avg_pooling_features = int(input_dim / 2 ** self.cnn_cells)
        self.final_out_channels = 512
        self.in_features_factor = int((self.global_avg_pooling_features) ** 2) if use_flatten else 1
        self.lin_in_features = self.final_out_channels * self.in_features_factor

        logging.info(f"Using {self.lin_in_features} features for the final linear layer input.")

        # define the model
        self.cnn = nn.Sequential(
            CNNCell(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1,
                    use_batch_norm=use_batch_norm, use_pool=True),
            CNNCell(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1,
                    use_batch_norm=use_batch_norm, use_pool=True),
            CNNCell(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1,
                    use_batch_norm=use_batch_norm, use_pool=True),
            CNNCell(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1,
                    use_batch_norm=use_batch_norm, use_pool=True),
            CNNCell(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1,
                    use_batch_norm=use_batch_norm, use_pool=True),
            nn.Flatten() if use_flatten else GlobalAveragePooling(self.global_avg_pooling_features),
            nn.Linear(in_features=self.lin_in_features, out_features=output_dim, bias=use_bias)
        )

    def forward(self, x):
        return self.cnn(x)