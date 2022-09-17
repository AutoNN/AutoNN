import torch  
from torch import nn

class LeNet(nn.Module):
    def __init__(self,in_channels=1,num_class = 10) -> None:
        super(LeNet,self).__init__()

        self.net=nn.Sequential( 
            nn.Conv2d(in_channels,out_channels=6,kernel_size=5),
            nn.AvgPool2d(2,2),
            nn.Tanh(),
            nn.Conv2d(6,out_channels=16,kernel_size=5),
            nn.AvgPool2d(2,2),
            nn.Tanh(),
            nn.Conv2d(16,out_channels=120,kernel_size=5)
        )
        self.fcs=nn.Sequential(
            nn.Linear(120,84),
            nn.Linear(84,num_class)
        )

    def forward(self,x):
        x = self.net(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fcs(x)
        return nn.functional.softmax(x,dim=1)

    




        
