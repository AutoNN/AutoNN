import torch
import torchvision as tv
from models.resnet import resnet
from pytorchsummary import summary
from torch.optim import Adam

def training(_model,LOSSfn,optimizer,trainloader):
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


def validation(model,testloader,loss):
    correct = 0
    total = 0
    Loss=0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _loss = loss(images,labels)
            Loss+=_loss
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Correct/Total :{correct}/{total}\t Accuracy:{correct/total *100}%')
        return Loss

def fit(Epochs,trainloader,validloader,model,criterion,optimizer):
    validlosses=[]
    trainlosses=[]
    for e in range(Epochs):
        trainloss = training(model,criterion,optimizer,trainloader)
        trainlosses.append(trainloss)
        valloss = training(model,validloader,criterion)
        validlosses.append(valloss)
        print(f'Epoch {e+1}/{Epochs} Training Loss:{trainloss} Validation Loss: {valloss} ')
    return trainlosses,validlosses 

def main(trainloader,validloader,criterion):
    all_models = [tv.models.efficientnet_b0(True),resnet(),tv.models.inception_v3(True)]

    all_val_loss=[]

    for model in all_models:
        optimizer = Adam(model.parameters(),lr=3e-4)
        trainloss,vallos = fit(2,trainloader,validloader,model,criterion,optimizer)
        all_val_loss.append(min(vallos))
    a = torch.tensor(all_val_loss)

    return all_models[torch.argmin(a,0)]

