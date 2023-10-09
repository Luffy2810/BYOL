import torch
import torch.nn as nn
from torchvision.models import resnet18
from collections import OrderedDict

class ResNet18(torch.nn.Module):
    
    def __init__(self):
        super(ResNet18, self).__init__()
        resnet = resnet18(weights=None)
        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.projection = MLPHead(in_channels=resnet.fc.in_features,mlp_hidden_size=512, projection_size=128)

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projection(h)


class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )
    def forward(self, x):
        return self.net(x)