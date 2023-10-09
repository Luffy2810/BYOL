import torch
import torch.nn as nn
from torchvision.models import resnet18
from collections import OrderedDict

def make_model():
    resnet = resnet18(weights=None)

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(resnet.fc.in_features, 100)),
        ('added_relu1', nn.ReLU(inplace=True)),
        ('fc2', nn.Linear(100, 50)),
        ('added_relu2', nn.ReLU(inplace=True)),
        ('fc3', nn.Linear(50, 25))
    ]))

    resnet.fc = classifier
    return resnet
