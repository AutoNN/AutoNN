# AutoNN.CNN.models.resnet.resnet



```python

def  resnet(architecture:int=18,**kwargs)->ResNet
```
This function can be used to call different Resnet architectures. All resnet architectures are implemted in `PyTorch` 

| Resnet | num_residual_block|
|:---:|:---:|
|  18  |  `[2,2,2,2]` |
|  34  |`  [3,4,6,3]` |
|   50 |  `[3,4,6,3]` |
|   101 |  ` [3,4,23,3]`|
|   152 |   `[3,8,36,3]`|
|   custom |   any combination you like  |



Example: 

To use ResNet50 with 10 output classes use:
```python    
from AutoNN.CNN.models.resnet import resnet

model = resnet(architecture=50,num_class=10)
print(model)
```
If you want to define your ResNet with custom number of blocks and then use:
```python
model = resnet(architecture=-1,in_channels=3,
            num_residual_block=[3,4,6,3],
            num_class=10,
            block_type='normal')
```


<span style="color:yellow">Arguments</span> :

- <span style="color:lime">architecture </span>= `-1` means you can choose the #layers of your resnet

- <span style="color:lime"> inchannels</span> = number of input channels

- <span style="color:lime"> num_residual_blocks</span> = number of `residual blocks` in each 
    layer, `[3,4,6,3]` means the model has 4 layers with

    
    - 1st layer containing 3 `residual blocks`, 
    - 2nd layer --> 4 `residual blocks`
    - 3rd layer --> 6 `residual blocks`
    - 4th layer --> 3 `residual blocks`

- <span style="color:lime">num_class</span> = number of classes (in this case it's 10)
- <span style="color:lime">block_type</span> = this has two types
  

     i) <span style="color:lime">'normal</span>' : has residual block with (3x3 conv,
                                        3x3 conv) in that order

    ii) <span style="color:lime">'botleneck</span>' : has limeisual block with (1x1 conv,
                                            3x3 conv,
                                            1x1 conv) in that order




<span style="color:yellow">Returns</span> :

`Resnet()` with desires number of layers and `residual blocks`


Example : Custom Model 

```python
    model = resnet(-1,num_residual_blocks=[2,1],num_class = 4,block_type='bottleneck')
```

??? output 
    ```
    ResNet(
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
    ```

    




