import torch.nn as nn
from inferno.extensions.layers.reshape import Flatten
from hyper_params import *

class ConvNet(nn.Module):
    def __init__(self, input_size, num_classes, mask):
        super(ConvNet, self).__init__()
        layers = []
        layers.append(nn.Conv2d(input_size, 64, kernel_size=3))
        layers.append(nn.Conv2d(64, 64, kernel_size=3))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        layers.append(Flatten())
        layers.append(nn.Linear(64*13*13, fc_size))
        layers.append(nn.Linear(fc_size, fc_size))
        layers.append(nn.Linear(fc_size, num_classes))
        self.layers = nn.Sequential(*layers)

        self.mask = mask

    def forward(self, x):
        if self.mask:
            for name, param in self.named_parameters():
                if 'weight' in name:
                    param.data = param.data * self.mask[name].float()
        out = self.layers(x)
        return out
