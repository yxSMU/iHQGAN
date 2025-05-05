import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(
        c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


class ClassicalGAN1():
    class Generator(nn.Module):
        def __init__(self, conv_dim=70):
            super().__init__()

            # encoding block
            self.conv1 = nn.Conv2d(1, conv_dim, kernel_size=16, stride=16, padding=0, bias=True)

            # decoding block
            self.deconv1 = nn.ConvTranspose2d(conv_dim, 1, kernel_size=16, stride=16, padding=0, bias=True)

        def forward(self, x):
            out_1 = F.leaky_relu(self.conv1(x), 0.2)  # (?, conv_dim, 1, 1)

            out = F.tanh(self.deconv1(out_1))  # (?, 1, 32, 32)

            return out

    
 class ClassicalCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(1024, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 1)

        def forward(self, x):
            x = x.view(x.shape[0], -1)
            x = F.leaky_relu(self.fc1(x), 0.2)
            x = F.leaky_relu(self.fc2(x), 0.2)
            return self.fc3(x)
 
