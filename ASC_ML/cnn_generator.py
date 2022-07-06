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

class ConvNet():
    def __init__(self,num_convlayers,batch_size,in_features,kernel_size,stride
                ):
        super(self,ConvNet).__innit__()
        