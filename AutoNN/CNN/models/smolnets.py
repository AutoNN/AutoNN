import torch  
from torch import nn

class SNet1(nn.Module):
    '''
    Total Trainable parameters 3,994
    '''
    def __init__(self,in_channels=3,num_class = 10,smaller=False):
        super(SNet1,self).__init__()
        self.smaller = smaller

        self.net=nn.Sequential( 
            nn.Conv2d(in_channels,out_channels=8,kernel_size=5),
            nn.AvgPool2d(2,2),
            nn.ReLU())
        
        if not smaller:
            self.mid = nn.Sequential(
                nn.Conv2d(8,out_channels=16,kernel_size=5),
                nn.AvgPool2d(2,2)
            )
            
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        if self.smaller:
            self.fcs = nn.Linear(8,num_class)
        else:
            self.fcs = nn.Linear(16,num_class)

    def forward(self,x):
        x = self.net(x)
        if not self.smaller:
            x=self.mid(x)

        x = self.avgpool(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fcs(x)
        return nn.functional.softmax(x,dim=1)

