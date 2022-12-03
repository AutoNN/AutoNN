# AutoNN.CNN.cnn_generator.CNN 

```python

class CNN(in_channels:int,numClasses:int,config: Optional[list[tuple]]=None):
```

## Parameters:

- <span style="color:violet">**in_channels**</span> :  The number of channels in an image

- <span style="color:violet">**numClass**</span> : Total number of classes in a classification problem

- <span style="color:violet">**config**</span> : Generated configuration by AutoNN's `CreateCNN.create_config()` method required to create a CNN model

## Methods:

- <span style="color:red">**summary()**</span> :
    
    Parameters:

    <span style="color:violet">**path**</span> :

    <span style="color:violet">**filename**</span> : Name of the model 
    
- <span style="color:red">**save()**</span>
- <span style="color:red">**load()**</span>
- <span style="color:red">**predict()**</span>
