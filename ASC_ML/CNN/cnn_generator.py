import random,torch
from torch import nn 
from .cnnBlocks import SkipLayer,Pooling
from typing import List,Tuple
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from tqdm import tqdm



def create_config(min,max)-> List[Tuple]:
    '''
    Args:
        min: minimum number of layers
        max: maximum number of layers
    Returns:
        List of cnn configuration
    example:

    >>> print(create_config(3,10))
    [('conv', 64, 64),
    ('pool', 1, 64),
    ('conv', 256, 512),
    ('conv', 64, 128),
    ('conv', 64, 64),
    ('pool', 0, 64)]

    '''
    L = random.randint(min,max)
    cfg = []
    
    f1 = 2**random.randint(4,9)
    f2 = 2**random.randint(4,9)
    cfg.append(('conv',f1,f2))    

    while (L-1):
        if random.random()<0.5:
            f1 = 2**random.randint(4,9)
            f2 = 2**random.randint(4,9)
            cfg.append(('conv',f1,f2))    
        if random.random()<0.5:
            if random.random() < 0.5:            
                cfg.append(('pool',1,f2))
            else:
                cfg.append(('pool',0,f2))
        L-=1
    return cfg
            
class CNN(nn.Module):
    def __init__(self,input_shape,numClasses,config) -> None:
        super(CNN,self).__init__()
        layers=[]
        for i in range(len(config)):
            if config[i][0]=='conv' and i==0:
                layers.append( SkipLayer(input_shape,config[i][1],config[i][2]))
            elif config[i][0]=='conv':
                layers.append( SkipLayer(config[i-1][2],config[i][1],config[i][2]))
            elif config[i][0]=='pool':
                if config[i][1]==1:
                    layers.append(Pooling('maxpool'))
                else:
                    layers.append(Pooling('avgpool'))
        self.network  = nn.Sequential(*layers)

        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.classifier = nn.Sequential(
            nn.Linear(config[-1][2],32),
            nn.ReLU(),
            nn.Linear(32,numClasses)
            )

    def forward(self,x):
        x = self.network(x)
        x = self.avgpool(x)
        x = torch.flatten(start_dim=1)
        x = self.classifier(x)
        return x


class Population:
    def __init__(self,_size:int,input_channels:int,num_classes:int) -> None:
        '''
        Args: 
            _size= population size
            input_channels = number of input channels

        '''
        self.size=_size
        self.ipshape = input_channels
        self.classes = num_classes
        self.popula  = [create_config(3,10) for _ in range(_size)]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.__create_cnns()

    def __create_cnns(self):
        self.cnns=[]
        for i in range(self.size):
            self.cnns.append(CNN(self.ipshape,self.classes,self.popula[i]))

    def print_all_architecture(self):
        for arch in self.cnns:
            print(arch)
            print('_'*150)
        
    def get_bestCNN(self,
                    path_trainset:str,
                    path_validationset:str=None,
                    path_testset:str=None,
                    split_required:bool=False,
                    batch_size:int=16,
                    lossFn:str='cross-entropy',
                    Optimizer:str='adam',
                    LR=3e-4
                    ):
        '''
        NOTE: make sure the path to your dataset is of 
            format 
        >>> "../dataset/train/"
            "../dataset/validation/"
            "../dataset/test/"

            i.e name your training image dataset folder
            as "train", validation set as "validation"
            and testing set as "test"

        Args:
            path_trainset: path to the training set
            path_validationset: path to the validation set
            path_testset: path to the test set
            split_required: default set to False 
                IF YOU ONLY HAVE TEH TRAINING DATA
                AND NOT VALIDATION SET AND TEST SET
                THEN set this to TRUE
        '''

        if lossFn == 'cross-entropy':
            criterion=nn.CrossEntropyLoss()
        


        if not split_required:

            trainSet = ImageFolder(path_trainset,transforms.Compose(
                [transforms.ToTensor()]
            ))
            
            validSet = ImageFolder(path_validationset,transforms.Compose(
                [transforms.ToTensor()]
            ))
            
            testSet = ImageFolder(path_testset,transforms.Compose(
                [transforms.ToTensor()]
            ))
        else:
            
            trainSet = ImageFolder(path_trainset,transforms.Compose(
                [transforms.ToTensor()]
            ))
            trainlen = int(len(trainSet)*0.7)
            testlen = len(trainSet) - trainlen # rest 30%
            validlen = int(testlen*0.5) #this is 50% of remaining 30%
            testlen -= validlen  #the rest 50%
            
            trainSet,validSet,testSet= random_split(trainSet,[trainlen,validlen,testlen])

            trainloader = DataLoader(trainSet,batch_size,shuffle=True,num_workers=4)
            valloader = DataLoader(validSet,batch_size,shuffle=True,num_workers=4)
            testloader = DataLoader(testSet,batch_size,shuffle=False,num_workers=4)

            history={}
            # test_LOSShistory =[]
            test_ACChistory =[]
            for i in range(self.size):
                if Optimizer =='adam':
                    optimizer= torch.optim.Adam(self.cnns[i].parameters(),lr=LR)
                train_performance=self.__training(self.cnns[i],trainloader,valloader,self.device,
                                        criterion,optimizer,epochs=10)
    
                history[f'cnn{i}']=train_performance

                test_performance = self.__test(self.cnns[i],testloader,self.device,criterion)
                # returns (loss,accuracy)
                # test_LOSShistory.append(test_performance[0])
                test_ACChistory.append(test_performance[1])
            
            best_accuracy,index = torch.max(torch.tensor(test_ACChistory))
            return best_accuracy, self.cnns[index]

    def __training(self,model,trainloader,validloader,device,LOSS,optimizer,epochs=10):
        performance={'trainloss':[],'trainacc':[],
                    'valloss':[],'valacc':[]}

        for _ in tqdm(range(epochs)):
            loss_per_epoch=0
            total=correct=0
            for x,y in trainloader:
                x,y = x.to(device),y.to(device)
                optimizer.zero_grad()
                yhat = model(x)
                loss = LOSS(y,yhat)
                loss.backward()
                optimizer.step()
                loss_per_epoch+=loss.item() 
                total +=y.size(0)
                pred=torch.max(yhat,dim=1)[1]
                correct += (pred==y).sum().item()

            performance['trainacc'].append(100* correct/total)
            performance['trainloss'].append(loss_per_epoch/len(trainloader))
            
            # for validaiton
            loss_=0
            total_=correct_=0
            with torch.no_grad():
                for x,y in validloader:
                    x,y = x.to(device),y.to(device)
                    yhat=model(x)
                    loss=LOSS(y,yhat)
                    loss_+=loss.item()
                    total_ +=y.size(0)
                    pred=torch.max(yhat,dim=1)[1]
                    correct_ += (pred==y).sum().item()
            performance['valloss'].append(loss_/len(validloader))
            performance['valacc'].append(100*correct_/total_)
    
        return performance


    def __test(self,model,loader,device,LOSS):
        loss_=0
        total_=correct_=0
        with torch.no_grad():
            for x,y in loader:
                x,y = x.to(device),y.to(device)
                yhat=model(x)
                loss=LOSS(y,yhat)
                loss_+=loss.item()
                total_ +=y.size(0)
                pred=torch.max(yhat,dim=1)[1]
                correct_ += (pred==y).sum().item()
        
        return (loss_/len(loader),100*correct_/total_)