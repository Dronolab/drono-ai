import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        # 128x128
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # 64x64
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # 32x32
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # 16x16
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # 8x8
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(
                16, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(
                32, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(
                64, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),

            # 64x64 -> 128x128
            nn.ConvTranspose2d(
                128, 256, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),

            # 128x128 -> 128x128
            nn.Conv2d(256, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded