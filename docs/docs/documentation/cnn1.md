# AutoNN.CNN.cnn_generator.CNN 

```python

class CNN(in_channels:int,numClasses:int,config: Optional[list[tuple]]=None):
```

## Parameters:

- <span style="color:violet">in_channels</span> :  The number of channels in an image

- <span style="color:violet">numClass</span> : Total number of classes in a classification problem

- <span style="color:violet">config</span> : Generated configuration by AutoNN's `CreateCNN.create_config()` method required to create a CNN model

## Methods:

- <span style="color:red">**summary()**</span> :
    
    Parameters:

    <span style="color:violet">input_shape</span> : Shape of the image `torch.tensor`  [C, H, W] 
    <pre>
        where,  C = number of channels
                H = height of the image
                W = width of the image
    </pre>
    
    <span style="color:violet">border</span> : `bool datatype` , default = `True` | It prints lines between layers while displaying the summary of a model if set to `True`   
    
- <span style="color:red">**save()**</span>

    <span style="color:violet">classes</span> : `List` of classes 

    <span style="color:violet">image_shape</span> : `Tuple[int, int]` (H,W) | dimension of the images 

    <span style="color:violet">path</span> : `Optional` | path to the directory where the model is intended to be stored 

    ??? note
        For the very first time `path` should be included, otherwise it will throw an `InvalidPathError` exception.

    <span style="color:violet">filename</span> : `str` | name of the model 

- <span style="color:red">**load()**</span>

    Parameters: 

   
    <span style="color:violet">PATH</span> : Path to the trained `model.pth`.

    <span style="color:violet">printmodel</span> : `bool datatype`, default = `False` |  print the model if set to `True`

    <span style="color:violet">loadmodel</span> :  `bool datatype`, default = `True` | loads all the stored weights if set to `True`
        
    
- <span style="color:red">**predict()**</span>

    <span style="color:violet">paths</span> : `Union[list | tuple]` | list of unknown images for testing or prediction 