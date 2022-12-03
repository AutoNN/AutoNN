# AutoNN.CNN.cnn_generator.CreateCNN


```python

class CreateCNN(_size:int=10):
```

<span style="color:yellow">Parameters</span>:

- <span style="color:violet">_size</span> : `int` | default = `10`| Maximum number of CNN models to be generated

## Methods:

### <span style="color:red">**create_config()**</span> :


This function will create configuration based on which CNN models will be generated.

<span style="color:yellow">Parameters</span>:

<span style="color:violet">min</span> : `int` | minimum number of layers the gnerated CNN model can have

<span style="color:violet">max</span> : `int` | maximum number of layers the gnerated CNN model can have


??? example
    ```python
    >>> print(create_config(3,10))
    >>> [('conv', 64, 64),
        ('pool', 1, 64),
        ('conv', 256, 512),
        ('conv', 64, 128),
        ('conv', 64, 64),
        ('pool', 0, 64)]
    ```

### <span style="color:red">**print_all_cnn_configs()**</span>  :

This function will print all the CNN architectures in PyTorch Format

<span style="color:yellow">Parameters</span>: `None`

### <span style="color:red">**print_all_architecture()**</span>

This function will print all the CNN architectures generated

<span style="color:yellow">Parameters</span>: `None` 
    
    
### <span style="color:red">**get_bestCNN()**</span>

<span style="color:yellow">
Parameters:    </span>

<span style="color:violet">path_trainset</span> : `str` | path to the image training set

<span style="color:violet">path_testset</span> : `str` | `Optional[str]` | path to the image test set

<span style="color:violet">split_required</span> : `bool` | default = `False` | set to true if only there is no test set

<span style="color:violet">batch_size</span> : `int` | default = `16` | Batch size 

<span style="color:violet">lossFn</span> : `str` | default = `crossentropy` | Most multiclass image classification problems use `CrossEntropyLoss`

<span style="color:violet">LR</span> : `float` | default = `3e4` | Learning Rate

<span style="color:violet">EPOCHS</span> : `int` | default = `10` | number Epochs

<span style="color:violet">image_shape</span> : `Tuple[int,int]` | default = `(28,28)` | dimension of the input image from the training dataset 

<span style="color:yellow">
Returns :</span>

Tuple containing the best model, it's accuracy and configuration

( best_CNN_model, best_model_config, history_of_all_models)