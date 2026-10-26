import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, in_channels=3, in_size=128):
        super().__init__()

        if in_size < 4 or (in_size & (in_size - 1)) != 0:
            raise ValueError(
                "in_size must be a power of 2 and >= 4"
            )

        num_layers = int(torch.log2(torch.tensor(in_size))) - 2

        channels = [
            min(256, 16 * 2 ** i)
            for i in range(num_layers)
        ]

        # Encoder
        encoder_layers = []
        current_channels = in_channels

        for out_channels in channels:
            encoder_layers.append(
                nn.Conv2d(
                    current_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            )
            encoder_layers.append(nn.ReLU())

            current_channels = out_channels

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []

        for i in range(len(channels) - 1, -1, -1):

            if i == 0:
                out_channels = in_channels
            else:
                out_channels = channels[i - 1]

            decoder_layers.append(
                nn.ConvTranspose2d(
                    current_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                )
            )

            if i == 0:
                decoder_layers.append(nn.Sigmoid())
            else:
                decoder_layers.append(nn.ReLU())

            current_channels = out_channels

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))