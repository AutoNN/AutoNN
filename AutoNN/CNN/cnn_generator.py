import random,torch,os
from torch import nn 
from numpy import argmax,array
from .cnnBlocks import SkipLayer,Pooling
from typing import List,Tuple,Any
from torchvision.datasets import ImageFolder
from torchvision import transforms
from pytorchsummary import summary as summ
from torch.utils.data import DataLoader,random_split
from tqdm import tqdm
from .models.resnet import resnet

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
    def __init__(self,in_channels,numClasses,config) -> None:
        super(CNN,self).__init__()
        layers=[]
        for i in range(len(config)):
            if config[i][0]=='conv' and i==0:
                layers.append( SkipLayer(in_channels,config[i][1],config[i][2]))
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
        x = torch.flatten(x,start_dim=1)
        x = self.classifier(x)
        return x

    def save(self,path='./best_models/',filename='Model.pth'):
        if not os.path.exists(path):
            os.makedirs(path)
        filename = os.path.join(path,filename)
        torch.save(self.state_dict(),filename)
        print(f'Model saved in directory: {path}{filename}')
    
    def summary(self,input_shape:tuple,border:bool=True):
        '''
        Args:
            input_shape = (num_channels,height,width)
            border = if set to false then it won't print
                lines between layers
        
        '''
        print(summ(input_shape,self,border))


class CreateCNN:
    def __init__(self,_size:int=10) -> None:
        '''
        Args: 
            _size= population size

        '''
        self.size=_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cnns=[]

    def print_all_cnn_configs(self): 
        for x,i in enumerate(self.configuration): 
            print(f'cnn{x} configuration:')
            print(i)
            print('_'*100)

    def print_all_architecture(self):
        '''
        This function will print all the 
        CNN architectures in PyTorch Format
        
        '''
        for arch in self.cnns:
            print(arch)
            print('_'*150)

    def __create_Cnns(self,len_dataset,input_shape,num_channels):
        l=len(input_shape)
        self.configuration  = []
        for _ in range(self.size):
            try:
                config_ = create_config(2,10)
                m1 = CNN(l,num_channels,config_)
                params, _,_ = summ(input_size =input_shape,model=m1,_print=False)
                print(params)
                if 0.7< params/len_dataset<1.5 :
                    self.cnns.append(m1)
                    self.configuration.append(config_)
            except Exception as e:
                # print(e)
                pass
        if not self.cnns:
            self.__create_Cnns(len_dataset,input_shape,num_channels)
        
        
    # def __meanNstd(self,path,loader=None):
    #     '''
    #     calculates the mean and std of an image dataset
    #     image has to be RGB image
    #     '''
    #     channelSum,channel2sum,batch = 0,0,0

    #     for X, _ in loader:
        
    #         channelSum += torch.mean(X,dim=[0,2,3])
    #         channel2sum += torch.mean(X**2,dim=[0,2,3])
    #         batch+=1
    #     mean= channelSum/batch
    #     std = (channel2sum/batch -mean**2)**0.5
        
    #     return mean,std


    def get_bestCNN(self,
                    path_trainset:str,
                    path_validationset:str=None,
                    path_testset:str=None,
                    split_required:bool=False,
                    batch_size:int=16,
                    lossFn:str='cross-entropy',
                    LR=3e-4,
                    EPOCHS=10,
                    image_shape:tuple=(28,28),
                    **kwargs
                    )->Tuple[float,Any,list,dict]:
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
            image_shape: (height,width) of the input image

            Optional Args:
                input_channels = number of input channels
                num_classes = Number of classes for classification
            
        Returns:
            Tuple containing the best model, it's accuracy and configuration
            (model_accuracy, CNN_model, model_config history_of_all_models)


        Example:
        >>> pop = CreateCNN(3,3,10) # first create an instance of the CreateCNN class 
            model,model_config,_ = pop.get_bestCNN('dataset',split_required=True)

        '''
        print(f'Default computing platform: {self.device}')


        if lossFn == 'cross-entropy':
            criterion=nn.CrossEntropyLoss()
        if not split_required:

            trainSet = ImageFolder(path_trainset,transforms.Compose(
                [transforms.ToTensor(),transforms.Resize(image_shape)]
            ))
            
            validSet = ImageFolder(path_validationset,transforms.Compose(
                [transforms.ToTensor(),transforms.Resize(image_shape)]
            ))
            
            testSet = ImageFolder(path_testset,transforms.Compose(
                [transforms.ToTensor(),transforms.Resize(image_shape)]
            ))
            len_classes = len(trainSet.classes)
            print('Classes: ',trainSet.classes)
        else:
            
            trainSet = ImageFolder(path_trainset,transforms.Compose(
                [transforms.ToTensor(),transforms.Resize(image_shape)]
            ))
            trainlen = int(len(trainSet)*0.7)
            testlen = len(trainSet) - trainlen # rest 30%
            validlen = int(testlen*0.5) #this is 50% of remaining 30%

            len_classes = len(trainSet.classes)
            print('Classes: ',trainSet.classes, '# Classes: ',len_classes)
            testlen -= validlen  #the rest 50%
            trainSet,validSet,testSet= random_split(trainSet,[trainlen,validlen,testlen])

        print(f'Training set size: {len(trainSet)} | Validation Set size: {len(validSet)} | Test Set size: {len(testSet)}')
        len_dataset = len(trainSet)
            
        # return len_dataset
        input_shape = tuple(trainSet[0][0].shape)
# ______________________________________________________________
        # self.val2 = kwargs.get('val2',"default value")
        self.inChannels = kwargs.get('in_channels',input_shape[0])
        self.numClasses = kwargs.get('num_classes',len_classes)
        if 4000<len_dataset < 10000:
            self.cnns.append(resnet(-1,in_channels=self.inChannels,num_residual_block=[0,1],num_class=self.numClasses))
            self.configuration.append('num_residual_block=[0,1] | resnet')
        elif len_dataset<=4000:
            raise Exception('''
            Your dataset size is too low!
            Use Augment.augment() function from CNN.utils.image_augmentation
            To augment your image dataset
            ''')
        else:
            self.__create_Cnns(len_dataset,input_shape,len_classes)
        print("Architecture search Complete..!")

# ______________________________________________________________


        trainloader = DataLoader(trainSet,batch_size,shuffle=True)
        valloader = DataLoader(validSet,batch_size,shuffle=True)
        testloader = DataLoader(testSet,batch_size,shuffle=False)

        history={}
        test_ACChistory =[]

        # optimizers 
        optims = []
        for i in range(len(self.cnns)):
            optims.append(torch.optim.Adam(self.cnns[i].parameters(),lr=LR))

        print('Searching for the best model. Please be patient. Thank you....')   

        print(f'Number of models generated: {len(self.cnns)}')

        for i in range(len(self.cnns)):
            print(f'Training CNN model cnn{i}')
            
            try:
                train_performance=self.__training(self.cnns[i],trainloader,valloader,self.device,
                                        criterion,optims[i],epochs=EPOCHS)
                history[f'cnn{i}']=train_performance
            except Exception as E:
                # print(f'2nd try er exception: {E}')
                pass
            
            print(f'Calculating test accuracy CNN model cnn{i}')
            try:
                test_performance = self.__test(self.cnns[i],testloader,self.device,criterion)
      
                test_ACChistory.append(test_performance[1])
            except Exception as E:
            
                # print(f'3rd try er exception: {E}')
                pass
            print('_'*150)
        
        if len(test_ACChistory)<1:
            self.__create_Cnns(len_dataset,input_shape,len_classes)
            
        index = argmax(array(test_ACChistory))
        print('Best test accuracy achieved by model cnn{index}: ',max(test_ACChistory))
        return  self.cnns[index],self.configuration[index],history

    def __training(self,model,trainloader,validloader,device,LOSS,optimizer,epochs):
        performance={'trainloss':[],'trainacc':[],
                    'valloss':[],'valacc':[]}
        model.train()
        for _ in range(epochs):
            print(f'Epoch: {_+1}')
            loss_per_epoch=0
            total=correct=0
            model= model.to(device)
            for x,y in tqdm(trainloader):
                # print(x.dtype,y.dtype)
                y = y.type(torch.LongTensor)   # casting to long

                x,y = x.to(device).float(),y.to(device)
                
                optimizer.zero_grad()
                yhat = model(x)
                # print(y.shape,yhat.shape)
                
                loss = LOSS(yhat,y)
                loss.backward()
                optimizer.step()
                loss_per_epoch+=loss.item() 
                total +=y.size(0)
                pred=torch.max(yhat,dim=1)[1]
                correct += (pred==y).sum().item()

            performance['trainacc'].append(100* correct/total)
            performance['trainloss'].append(loss_per_epoch/len(trainloader))
            print(f'Training Accuracy: {performance["trainacc"][_]}\t Training Loss:{performance["trainloss"][_]}')
            
            # for val        idaiton
            model.eval()
            loss_=0
            total_=correct_=0
            model= model.to(device)
            with torch.no_grad():
                for x,y in tqdm(validloader):
                    y = y.type(torch.LongTensor)   # casting to long
                    x,y = x.to(device).float(),y.to(device)
                    yhat=model(x)
                    loss=LOSS(yhat,y)
                    loss_+=loss.item()
                    total_ +=y.size(0)
                    pred=torch.max(yhat,dim=1)[1]
                    correct_ += (pred==y).sum().item()
            performance['valloss'].append(loss_/len(validloader))
            performance['valacc'].append(100*correct_/total_)

            if _>0:
                if performance['valloss'][_]<performance['valloss'][_-1]:
                    torch.save(model.state_dict(),'temp.pth')

            print(f'Validation Accuracy: {performance["valacc"][_]}\t Validation Loss:{performance["valloss"][_]}')
        model.load_state_dict(torch.load('temp.pth'))    
        return performance


    def __test(self,model,loader,device,LOSS):
        model.eval()
        loss_=0
        total_=correct_=0
        model= model.to(device)
        with torch.no_grad():
            for x,y in loader:
                y = y.type(torch.LongTensor)   # casting to long
                x,y = x.to(device),y.to(device)
                yhat=model(x)
                loss=LOSS(yhat,y)
                loss_+=loss.item()
                total_ +=y.size(0)
                pred=torch.max(yhat,dim=1)[1]
                correct_ += (pred==y).sum().item()
        print(f'Test ACCuracy: {100*correct_/total_}\t Test Loss: {loss_/len(loader)}')
        print('-'*150)
        return (loss_/len(loader),100*correct_/total_)