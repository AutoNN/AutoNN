from typing import List,Dict,Any,Optional
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch import nn 
from models.resnet import resnet
from pytorchsummary import summary
from torch.optim import Adam

from torchvision import models


# Typing alias
Models = List[nn.Module]


cnnmodel = resnet(18)
cnnmodel.resnet = cnnmodel.resnet[:4]
print(cnnmodel)


def training(_model,LOSSfn,optimizer,trainloader:DataLoader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_per_epoch=0
    for x,y in trainloader:
        x,y = x.to(device),y.to(device)
        yhat = _model(x)
        loss = LOSSfn(y,yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_per_epoch+=loss 
    loss_per_epoch/=len(trainloader)
    return loss_per_epoch


def best_model(models:Models):

    pass
    