from typing import List,Dict,Any,Optional
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch import nn 
from models.resnet import resnet
from torchsummary import summary
from torch.optim import Adam
from __init__ import device as DEVICE

from torchvision import models


# Typing alias
Models = List[nn.Module]


cnnmodel = resnet(18)
cnnmodel.resnet = cnnmodel.resnet[:4]
print(cnnmodel)

def training(_model,optimizer,trainloader:DataLoader,device:Optional[torch.device]=DEVICE):
    
    for x,y in trainloader:
        x,y = x.to(device),y.to(device)


def best_model(models:Models):

    pass
    