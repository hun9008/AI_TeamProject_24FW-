import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms

from torchinfo import summary

import timm

swin = timm.create_model('swin_base_patch4_window7_224', pretrained=True)

print(summary(swin, input_size=(32, 3, 224, 224)))