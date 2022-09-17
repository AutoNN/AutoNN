## How to use

```python
from AutoNN.CNN.cnn_generator import CreateCNN
pop = CreateCNN(3,3,10)
best_acc,model,bestconfig,history = pop.get_bestCNN('dataset',split_required=True,EPOCHS=1)
```
### Output:
```
Classes:  ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
Training set size: 42000 | Validation Set size: 9000 | Test Set size: 9000
Training CNN model cnn0
Epoch: 1
Training Accuracy: 94.97380952380952     Training Loss:0.19278352837641502
Validation Accuracy: 98.4        Validation Loss:0.05288387521883046      
Calculating test accuracy CNN model cnn0
Test ACCuracy: 98.36666666666666         Test Loss: 0.058304798307946056
------------------------------------------------------------------------------------------------------------------------------------------------------
______________________________________________________________________________________________________________________________________________________
Training CNN model cnn1
Epoch: 1
Training Accuracy: 95.71904761904761     Training Loss:0.160947915122046    
Validation Accuracy: 97.71111111111111   Validation Loss:0.07768459637404533
Calculating test accuracy CNN model cnn1
Test ACCuracy: 97.56666666666666         Test Loss: 0.07887562570000005
------------------------------------------------------------------------------------------------------------------------------------------------------
```

# pop.print_all_cnn_configs()
```python
model.save()
model.summary((3,28,28))
```

```
Model saved in directory: ./best_models/
               Layer    Output Shape                Kernal Shape        #params                 #(weights + bias)       requires_grad
------------------------------------------------------------------------------------------------------------------------------------------------------   
            Conv2d-1    [1, 512, 28, 28]           [512, 3, 3, 3]       14336                   (13824 + 512)           True True 
       BatchNorm2d-2    [1, 512, 28, 28]               [512]            1024                    (512 + 512)             True True 
              ReLU-3    [1, 512, 28, 28]                                                                                
            Conv2d-4    [1, 128, 28, 28]          [128, 512, 3, 3]      589952                  (589824 + 128)          True True 
       BatchNorm2d-5    [1, 128, 28, 28]               [128]            256                     (128 + 128)             True True 
            Conv2d-6    [1, 128, 28, 28]           [128, 3, 1, 1]       512                     (384 + 128)             True True 
              ReLU-7    [1, 128, 28, 28]
         AvgPool2d-8    [1, 128, 14, 14]
            Conv2d-9    [1, 256, 14, 14]          [256, 128, 3, 3]      295168                  (294912 + 256)          True True
      BatchNorm2d-10    [1, 256, 14, 14]               [256]            512                     (256 + 256)             True True
             ReLU-11    [1, 256, 14, 14]
           Conv2d-12    [1, 64, 14, 14]           [64, 256, 3, 3]       147520                  (147456 + 64)           True True
      BatchNorm2d-13    [1, 64, 14, 14]                 [64]            128                     (64 + 64)               True True
           Conv2d-14    [1, 64, 14, 14]           [64, 128, 1, 1]       8256                    (8192 + 64)             True True
             ReLU-15    [1, 64, 14, 14]
           Conv2d-16    [1, 128, 14, 14]          [128, 64, 3, 3]       73856                   (73728 + 128)           True True
      BatchNorm2d-17    [1, 128, 14, 14]               [128]            256                     (128 + 128)             True True
             ReLU-18    [1, 128, 14, 14]
           Conv2d-19    [1, 32, 14, 14]           [32, 128, 3, 3]       36896                   (36864 + 32)            True True
      BatchNorm2d-20    [1, 32, 14, 14]                 [32]            64                      (32 + 32)               True True
           Conv2d-21    [1, 32, 14, 14]            [32, 64, 1, 1]       2080                    (2048 + 32)             True True
             ReLU-22    [1, 32, 14, 14]
        AvgPool2d-23    [1, 32, 7, 7]
           Conv2d-24    [1, 128, 7, 7]            [128, 32, 3, 3]       36992                   (36864 + 128)           True True
      BatchNorm2d-25    [1, 128, 7, 7]                 [128]            256                     (128 + 128)             True True
             ReLU-26    [1, 128, 7, 7]
           Conv2d-27    [1, 128, 7, 7]            [128, 128, 3, 3]      147584                  (147456 + 128)          True True
      BatchNorm2d-28    [1, 128, 7, 7]                 [128]            256                     (128 + 128)             True True
           Conv2d-29    [1, 128, 7, 7]            [128, 32, 1, 1]       4224                    (4096 + 128)            True True
             ReLU-30    [1, 128, 7, 7]
        AvgPool2d-31    [1, 128, 3, 3]
AdaptiveAvgPool2d-32    [1, 128, 1, 1]
           Linear-33    [1, 32]                      [32, 128]          4128                    (4096 + 32)             True True
             ReLU-34    [1, 32]
           Linear-35    [1, 10]                       [10, 32]          330                     (320 + 10)              True True
______________________________________________________________________________________________________________________________________________________
Total parameters 1,364,586
Total Non-Trainable parameters 0
Total Trainable parameters 1,364,586
(1364586, 1364586, 0)
```


```python
print(model)
print(best_acc)
print(f'best config {bestconfig}')
print(f'history {history}')
```

```
CNN(
  (network): Sequential(
    (0): SkipLayer(
      (skiplayers): Sequential(
        (0): Conv2d(3, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (skip_connection): Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1))
      (relu): ReLU()
    )
    (1): Pooling(
      (pool): AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
    )
    (2): SkipLayer(
      (skiplayers): Sequential(
        (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (skip_connection): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
      (relu): ReLU()
    )
    (3): SkipLayer(
      (skiplayers): Sequential(
        (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (skip_connection): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
      (relu): ReLU()
    )
    (4): Pooling(
      (pool): AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
    )
    (5): SkipLayer(
      (skiplayers): Sequential(
        (0): Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (skip_connection): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
      (relu): ReLU()
    )
    (6): Pooling(
      (pool): AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (classifier): Sequential(
    (0): Linear(in_features=128, out_features=32, bias=True)
    (1): ReLU()
    (2): Linear(in_features=32, out_features=10, bias=True)
  )
)

98.36666666666666

best config [('conv', 512, 128), ('pool', 0, 128), ('conv', 256, 64), ('conv', 128, 32), ('pool', 0, 32), ('conv', 128, 128), ('pool', 0, 128)]       

history {'cnn0': {'trainloss': [0.19278352837641502], 'trainacc': [94.97380952380952], 'valloss': [0.05288387521883046], 'valacc': [98.4]}, 'cnn1': {'trainloss': [0.160947915122046], 'trainacc': [95.71904761904761], 'valloss': [0.07768459637404533], 'valacc': [97.71111111111111]}}
```

```python
from ASC_ML.CNN.cnn_generator import CNN
PATH='./best_models/model.pth' # path to your saved model
model = CNN(in_channels=3,numClasses=10,config = bestconfig)
model.load_state_dict(torch.load(PATH))
model.eval() #setting the model into evaluation mode 
# now model is ready for prediction task
```
