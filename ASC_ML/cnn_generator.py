import torch
from torch import nn 
from torchvision import models 
from torchsummary import summary

'''
General models: Efficent Net (B0 - B7)
                Alex Net 
                Dense Net 
                VGG 

'''

class ConvNet(nn.Module):

    def __init__(self,num_convlayers,batch_size,in_features):
        super(self,ConvNet).__init__()

        self.l1 = nn.Sequential(
            nn.Conv2d(in_features,32,5), # n-4
            nn.ReLU(),
            nn.MaxPool2d(2,2), # n-4//2

        )
        self.l2 = nn.Sequential(
            nn.Conv2d(32,32,3), # n/2 -4
            nn.ReLU(),
            nn.MaxPool2d(2,2), # n/4 -2
        )

        self.l3 = nn.Sequential(
            nn.Conv2d(32,64,5), 
            nn.ReLU())
    
    def forward(self,x):
        
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x  
