import torch  
from torch import nn

class LeNet(nn.Module):
    def __init__(self,in_channels=1,num_class = 10,activation='tanh') -> None:
        super(LeNet,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,out_channels=6,kernel_size=5)
        self.conv2 = nn.Conv2d(6,out_channels=16,kernel_size=5)
        self.conv3 = nn.Conv2d(16,out_channels=120,kernel_size=5)
        self.avgpool = nn.AvgPool2d(2,2)
        self.fc1 = nn.Linear(120,84)
        self.fc2 = nn.Linear(84,num_class)

        if activation.lower()=='tanh':
            self.acti = nn.Tanh()
        elif activation.lower()=='sigmoid':
            self.acti = nn.Sigmoid()
        else:
            self.acti = nn.ReLU()
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.acti(self.avgpool(x))
        x = self.conv2(x)
        x = self.acti(self.avgpool(x))
        x = self.conv3(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fc2(self.fc1(x))
        return nn.functional.softmax(x)

    




        


# def test():
#     from torchsummary import summary
#     model = LeNet()
#     print(model)
#     print('.______________________________________.')
#     print(summary(model.cuda(),(1,32,32)))
# test()