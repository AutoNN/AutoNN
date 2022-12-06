# AutoNN.CNN.utils.image_augmentation.Augment

```python

class Augment(path)
```
This class will augment the image dataset using the following image operations: 

- Rotations
- Horizontal Flip
- Vertical Flip


<span style="color:yellow">Parameters</span> :

path : `str` |  path to the image dataset folder

??? tip 
        '''
            path: provide the path to your image folder
            which you want to augment
            ../Folder/dataset/cats/x1.png
            ../Folder/dataset/cats/x2.png
            .
            .
            .
            ../Folder/dataset/dogs/xx1.png
            ../Folder/dataset/dogs/xx2.png
            ../Folder/dataset/dogs/xx3.png
            .
            .
            path = '../Folder/dataset/'
        '''

## Method:

### <span style="color:red">**augment()**</span> :
Call this function to start augmentation

<span style="color:yellow">Parameters</span>: `None`

<span style="color:yellow">Returns</span>: `None`