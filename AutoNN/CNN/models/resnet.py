import torch,json,os 
from torch import nn 
from AutoNN.exceptions import InvalidPathError
from torchvision import transforms 
from PIL import Image
from pytorchsummary import summary as summ

PATHJSON = os.path.dirname(__file__).removesuffix('\CNN\models')
PATHJSON= os.path.join(PATHJSON,'default_config.json')

class BasicBlock(nn.Module):
    def __init__(self,in_features=64,out_features=64,stride=[1,1],down_sample=False):
        super(BasicBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_features,out_features,3,stride[0],padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_features,out_features,3,stride[1],padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_features)

        self.down_sample = down_sample
        if down_sample:
            self.downsample = nn.Sequential(
                    nn.Conv2d(in_features,out_features,1,2,bias=False),
                    nn.BatchNorm2d(out_features)
                )

    def forward(self,x):
        x0=x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.down_sample:
            x0 = self.downsample(x0)
        x = x + x0
        x= self.relu(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self,inFeatures=64,outFeatures=64,kSize=[1,3,1],stride=[1,2,1],
    dn_sample=False,dnSample_stride=1) -> None:
        super(Bottleneck,self).__init__()
        
        
        self.conv1 = nn.Conv2d(inFeatures,outFeatures,kSize[0],stride[0],bias=False)
        self.bn1 = nn.BatchNorm2d(outFeatures)
        self.conv2 = nn.Conv2d(outFeatures,outFeatures,kSize[1],stride[1],padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(outFeatures)
        self.conv3 = nn.Conv2d(outFeatures,outFeatures*4,kSize[2],stride[2],bias=False)
        self.bn3 = nn.BatchNorm2d(outFeatures*4)
        self.relu = nn.ReLU(True)
        

        self.ds = dn_sample
        if dn_sample:
            self.downSample = nn.Sequential(
                nn.Conv2d(inFeatures,outFeatures*4,1,stride=dnSample_stride,bias=False),
                nn.BatchNorm2d(outFeatures*4)            
            )
        
    
    def forward(self,x):
        x0 = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        if self.ds:
            x0 = self.downSample(x0)
        x = x+x0
        return x

class ResNet(nn.Module):
    """
    Resnet 50 architecture is set as default architecture.
    To use Resnet34 or ResNet18 
    change block_type to 'normal' (set as default)

    num_residual_block = [3,4,6,3] | representes the number of blocks/bottlenecks
    in each layers 

    To change number of classes use the following line of code:
    >>>model = ResNet(num_class=5)
    
    """
    def __init__(self,in_channels=3,num_residual_block=[3,4,6,3],num_class=10,block_type='normal'):
        super(ResNet,self).__init__()
        self.__ig=None 
        self.__classes=None

        self.conv1 = nn.Conv2d(in_channels,64,7,2,3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(3,2,1)

        if block_type.lower() == 'bottleneck':    
            self.resnet,outchannels = self.__bottlenecks(num_residual_block)
        else:
            self.resnet,outchannels = self.__layers(num_residual_block)
    

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_features=outchannels,out_features=num_class,bias=True)

        
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.resnet(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x 
    
    def __layers(self,num_residual_block):
        layer=[]
        layer += [BasicBlock()]*num_residual_block[0]
        inchannels=64
        for numOFlayers in num_residual_block[1:]:
            stride = [2,1]
            downsample=True
            outchannels = inchannels*2
            for _ in range(numOFlayers):
                layer.append(BasicBlock(inchannels,outchannels,stride,down_sample=downsample))
                inchannels = outchannels
                downsample = False 
                stride=[1,1]
            
        return nn.Sequential(*layer),outchannels

    
    def __bottlenecks(self,numres):
        
        layer=[]
        
        stride = [1,1,1]
        dnStride=1
        inchan = 64
        for i,numOFlayers in enumerate(numres):
            dn_sample = True
            outchan = 64*(2**i)

            for _ in range(numOFlayers):
                layer+=[ 
                    Bottleneck(inchan,outchan,stride=stride,
                    dn_sample=dn_sample,dnSample_stride=dnStride)
                ]
                inchan = outchan*4
                dn_sample = False
                stride = [1,1,1]   
            dn_sample=True 
            stride = [1,2,1]
            dnStride=2
            

        return nn.Sequential(*layer),inchan

    def save(self,**kwargs):
        # self.val2 = kwargs.get('val2',"default value")
        path = kwargs.get('path',None)
        filename = kwargs.get('filename','resnet0')
        image_shape = kwargs.get('image_shape')
        classes = kwargs.get('classes')

        with open(PATHJSON) as f:
            data = json.load(f)
        if data['path_cnn_models']:
            path = data['path_cnn_models']
        else: 
            if not path:
                raise InvalidPathError
            data['path_cnn_models'] = path

        with open(PATHJSON, "w") as f:
            json.dump(data, f)  
        if not os.path.exists(path):
            os.makedirs(path)
        
        torch.save(self,os.path.join(path,f'{filename}.pth'))
        print(f'Model saved in directory: {path}')

        with open(os.path.join(path,f'{filename}.json'),'w') as f:
            d = {"classes":classes,"image shape":image_shape}
            json.dump(d, f)

    def load_model(self,PATH,**kwargs):

        configfile = os.path.split(PATH)[-1].replace('.pth','.json')
        with open(os.path.join(os.path.split(PATH)[0],configfile),'r') as f:
            data = json.load(f)
            
            self.__classes = data['classes']
            self.__ig = data['image shape']

        # self.load_state_dict(torch.load(PATH))
        self = torch.load(PATH)
        self.eval()
        print('Resnet Model Loaded')

    def predict(self,paths):
        transform = transforms.Compose([transforms.ToTensor(),
                transforms.Resize(self.__ig)])
        
        preds=list()
        for img in paths:
            image = Image.open(img)
            x = transform(image).float()
            x = x.unsqueeze(0)
            output = self.forward(x)
            preds.append(self.__classes[torch.argmax(output,1).item()])
        
        return preds

    def summary(self,input_shape,**kwargs):
        print(summ(input_shape,**kwargs))
        

def  resnet(architecture:int=18,**kwargs)->ResNet:
    '''
    Args:
    architecture: choose the type of ResNet you want to use
    example: 
        To use ResNet50 with 10 output classes use:
    >>> model = resnet(architecture=50,num_class=10)
    
    if you want to define your ResNet with custom number of blocks and then use:
    >>> model = resnet(architecture=-1,in_channels=3,
                num_residual_block=[3,4,6,3],
                num_class=10,
                block_type='normal')

    Explanation: 
    Arguments:  architecture = -1 means you can choose the #layers of your resnet
                inchannels = number of input channels
                num_residual_blocks = number of residual blocks in each 
                    layer, `[3,4,6,3]` means the model has 4 layers with
                    1st layer containing 3 residual blocks, 
                    2nd layer --> 4 residual blocks
                    3rd layer --> 6 residual blocks
                    4th layer --> 3 residual blocks
                num_class = number of classes (in this case it's 10)
                block_type = this has two types
                    i) 'normal'
                    ii) 'bottleneck'

                    'normal' has residual block with (3x3 conv,
                                                      3x3 conv) in that order
                    'botleneck' has redisual block with (1x1 conv,
                                                         3x3 conv,
                                                         1x1 conv) in that order
    

    Returns: Resnet() with desires number of layers and residual blocks

    example: 
    >>> model = resnet(-1,num_residual_blocks=[2,1],num_class = 4,bllock_type='bottleneck')
    
    output: ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (resnet): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (downSample): Sequential(
        (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (2): Bottleneck(
      (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (downSample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=512, out_features=4, bias=True)
)

    '''
    if architecture ==18:
        return ResNet(num_residual_block=[2,2,2,2],**kwargs)
    elif architecture ==34:
        return ResNet(num_residual_block=[3,4,6,3],**kwargs)
    elif architecture ==50:
        return ResNet(num_residual_block=[3,4,6,3],block_type='bottleneck',**kwargs)
    elif architecture ==101:
        return ResNet(num_residual_block=[3,4,23,3],block_type='bottleneck',**kwargs)
    elif architecture ==152:
        return ResNet(num_residual_block=[3,8,36,3],block_type='bottleneck',**kwargs)
    else:
        return ResNet(**kwargs)


