from torch import nn
import torch.optim as optim
from tqdm import tqdm 
import torch 
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

class Autoencoders(nn.Module):
    def __init__(self,num_classes,in_channels=3):
        '''
        
        '''
        super(Autoencoders, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels,16,5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(16,32,3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
        # if self.__train:
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32,16,4,stride=2,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,16,3,stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,in_channels,5,stride=2,output_padding=1),
            nn.Sigmoid()

        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(2,2),
            nn.Linear(in_channels,128),
            nn.ReLU(),
            nn.Linear(128,num_classes))
        
        
    def forward(self,x):
        latent_space = self.encoder(x)
        x = self.classifier(latent_space)

        img = self.decoder(latent_space)
        return x,latent_space,img


class TrainAutoens:
    def __init__(self,num_classes,
                in_channels,
                trainset_path,
                testset_path,
                batch_size=4,
                lr1=0.001,
                lr=3e-4) -> None:

        trainset = ImageFolder(trainset_path,transforms.Compose([transforms.ToTensor()]))
        testset = ImageFolder(testset_path,transforms.Compose([transforms.ToTensor()]))
        
        self.trainloader = DataLoader(trainset,batch_size,shuffle=True)
        self.testloader = DataLoader(testset,batch_size,shuffle=False)

        self.model = Autoencoders(num_classes,in_channels)
        self.reconstructionLoss = nn.MSELoss()
        self.criterion = nn.CrossEntropyLoss()
        self.parameters = list()
        self.parameters.extend(self.model.encoder.parameters())
        self.parameters.extend(self.model.decoder.parameters())
        self.adam1 = optim.Adam(self.parameters,lr=lr1)
        self.adam2 =optim.Adam(self.model.classifier.parameters(),lr=lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __train(self,epoch):
        self.model.train()
        total_loss = 0
        total_loss1 = 0
        print(f"Epoch {epoch+1}")
        
        for images in tqdm(self.trainloader):
            self.adam1.zero_grad()
            self.adam2.zero_grad()
            
            images,Ys = images
            images = images.to(self.device)
            Ys = Ys.to(self.device)
            yhat,_, out = self.model(images)
            
            loss = self.reconstructionLoss(out, images) # l2 norm
            loss1 = self.criterion(yhat, Ys) # cross entropy
            
            l = loss+loss1
            
            l.backward() 
            total_loss+=l
            total_loss1+=loss1
            self.adam1.step()
            self.adam2.step()
        
        return total_loss/len(self.trainloader),total_loss1/len(self.trainloader)

    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images,labels in tqdm(self.testloader):
                images,labels = images.to(self.device), labels.to(self.device)
                
                yhat,_,_=self.model(images)
                    
                _,predicted = torch.max(yhat.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        
        print(f'Accuracy of the network on the {total} test images: {100 * correct / total} % {correct}/{total}')

    def fit(self,epochs=10):
        preds=list()
        for _ in range(epochs):
            __,p=self.__train(_)
            
            preds.append(p)
        
        print('Training Complete!')

        fig = plt.figure(figsize=(7,5))
        plt.plot(range(epochs),preds,'g')
        plt.xlabel('Epochs')
        plt.ylabel('Prediction loss')
        plt.show()
        pass 