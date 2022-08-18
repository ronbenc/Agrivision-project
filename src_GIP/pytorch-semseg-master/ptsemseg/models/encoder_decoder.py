
"""""
Adapted from:

"""

import torch
import torch.nn as nn


class encoder_decoder(nn.Module):
    def __init__(self, n_classes=17, n_channels=3):
        super().__init__()
        n_maps = [8,16,32,64]

        self.encoder = nn.Sequential(
            # torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),

            nn.Conv2d(n_channels, n_maps[0], kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(n_maps[0]),
            nn.LeakyReLU(True),
            nn.Conv2d(n_maps[0], n_maps[1], kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(n_maps[1]),
            nn.LeakyReLU(True),
            nn.Conv2d(n_maps[1], n_maps[2], kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(n_maps[2]),
            nn.LeakyReLU(True),
            nn.Conv2d(n_maps[2], n_maps[3], kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(n_maps[3]),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(n_maps[3], n_maps[2], kernel_size=3, stride=1, padding=0, output_padding=0),
            nn.BatchNorm2d(n_maps[2]),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(n_maps[2], n_maps[1], kernel_size=3, stride=1, padding=0, output_padding=0),
            nn.BatchNorm2d(n_maps[1]),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(n_maps[1], n_maps[0], kernel_size=3, stride=1, padding=0, output_padding=0),
            nn.BatchNorm2d(n_maps[0]),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(n_maps[0], n_classes, kernel_size=3, stride=1 ,padding=0, output_padding=0),
            nn.BatchNorm2d(n_classes),
            nn.Softmax(1)
        )

    # self.encoder = nn.Sequential(
    #     # torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
    #
    #     nn.Conv2d(n_channels, n_maps[0], kernel_size=5, stride=1, padding=0),
    #     nn.BatchNorm2d(n_maps[0]),
    #     nn.LeakyReLU(True),
    #     # nn.MaxPool2d(2), # shape: batch, 16, 254, 254
    #     nn.Conv2d(n_maps[0], n_maps[1], kernel_size=7, stride=1, padding=0),
    #     nn.BatchNorm2d(n_maps[1]),
    #     nn.LeakyReLU(True),
    #     nn.Conv2d(n_maps[1], n_maps[2], kernel_size=7, stride=1, padding=0),
    #     nn.BatchNorm2d(n_maps[2]),
    # )
    #
    # self.decoder = nn.Sequential(
    #     nn.ConvTranspose2d(n_maps[2], n_maps[1], kernel_size=7, stride=1, output_padding=0),
    #     nn.BatchNorm2d(n_maps[1]),
    #     nn.LeakyReLU(True),
    #     nn.ConvTranspose2d(n_maps[1], n_maps[0], kernel_size=7, stride=1),
    #     nn.BatchNorm2d(n_maps[0]),
    #     nn.LeakyReLU(True),
    #     nn.ConvTranspose2d(n_maps[0], n_classes, kernel_size=5, stride=1),
    #     nn.BatchNorm2d(n_classes),
    #     nn.Softmax(1)
    # )


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x