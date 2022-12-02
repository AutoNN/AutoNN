# CNN Models


AutoNN's CNN (Convolutional Neural Network) Models are built on PyTorch.


## 1. How to use

Import `CreateCNN` from CNN's `cnn_generator` module


```python
from AutoNN.CNN.cnn_generator import CreateCNN

obj = CreateCNN() # create a CreateCNN object

model,bestconfig,history =\
     obj.get_bestCNN('D:/kaggle/APPLE_DISEASE_DATASET',
     split_required=True,batch_size=4,EPOCHS=5,image_shape=(300,300))
# call the object's get_bestCNN() function
```


 Output:
```python
# print the values to see the results
print(model)
print(f'best config {bestconfig}')
print(f'history {history}')
```
And here we go:
```


░█▀▀█ █░░█ ▀▀█▀▀ █▀▀█ ▒█▄░▒█ ▒█▄░▒█
▒█▄▄█ █░░█ ░░█░░ █░░█ ▒█▒█▒█ ▒█▒█▒█
▒█░▒█ ░▀▀▀ ░░▀░░ ▀▀▀▀ ▒█░░▀█ ▒█░░▀█

An AutoML framework by
Anish Konar, Arjun Ghosh, Rajarshi Banerjee, Sagnik Nayak.

Training set size: 18685 | Validation Set size: 4004 | Test Set size: 4004
Architecture search Complete..! Time Taken:  0:01:17.244049
Searching for the best model. Please be patient. Thank you....
Number of models generated: 2
Training CNN model cnn0
Epoch: 1
100%|█████████████████| 4672/4672 [17:36<00:00,  4.42it/s]
Training Accuracy: 52.49130318437249     Training Loss:1.0508242789209399                                                                            
100%|█████████████████| 1001/1001 [03:07<00:00 35it/]                                                                                                               
Validation Accuracy: 56.64335664335665   Validation Loss:0.9246639279457001
Epoch: 2
100%|█████████████████| 4672/4672 [08:55<00:00,  8.72it/s]         
Training Accuracy: 60.28900187316029     Training Loss:0.8835673358543354
100%|█████████████████| 1001/1001 [01:14<00:00, 13.50it/s]
Validation Accuracy: 59.96503496503497   Validation Loss:0.8518805852630636
Epoch: 3
100%|███████████████████| 4672/4672 [08:34<00:00,  9.08it/s] 
Training Accuracy: 63.60717152796361     Training Loss:0.809855165998595
100%|███████████████████| 1001/1001 [01:08<00:00, 14.54it/s] 
Validation Accuracy: 63.51148851148851   Validation Loss:0.7923010227444408
Epoch: 4
100%|███████████████████| 4672/4672 [08:44<00:00,  8.91it/s] 
Training Accuracy: 65.99411292480599     Training Loss:0.7608422355639448
100%|███████████████████| 1001/1001 [01:12<00:00, 13.72it/s] 
Validation Accuracy: 64.83516483516483   Validation Loss:0.7738665017810139
Epoch: 5
100%|███████████████████| 4672/4672 [09:14<00:00,  8.42it/s] 
Training Accuracy: 68.32218356970833     Training Loss:0.7230342619339553
100%|███████████████████| 1001/1001 [01:10<00:00, 14.14it/s]
Validation Accuracy: 62.06293706293706   Validation Loss:0.8454483170899656
Calculating test accuracy CNN model cnn0
Test ACCuracy: 65.23476523476523         Test Loss: 0.7666534855143055
------------------------------------------------------------------------------------------------------------------------------------------------------
______________________________________________________________________________________________________________________________________________________
Training CNN model cnn1
Epoch: 1
100%|███████████████████| 4672/4672 [08:36<00:00,  9.04it/s]
Training Accuracy: 52.66256355365266     Training Loss:1.0577738225932092
100%|███████████████████| 1001/1001 [00:56<00:00, 17.59it/s]
Validation Accuracy: 58.56643356643357   Validation Loss:0.9101097066919287
Epoch: 2
100%|███████████████████| 4672/4672 [07:32<00:00, 10.31it/s]
Training Accuracy: 60.81883864062082     Training Loss:0.8753257903688243
100%|███████████████████| 1001/1001 [01:24<00:00, 11.91it/s]
Validation Accuracy: 62.837162837162836  Validation Loss:0.8394484582629713
Epoch: 3
100%|███████████████████| 4672/4672 [07:08<00:00, 10.90it/s]
Training Accuracy: 64.23334225314423     Training Loss:0.8081053741330485
100%|███████████████████| 1001/1001 [00:54<00:00, 18.22it/s]
Validation Accuracy: 66.4085914085914    Validation Loss:0.7697237971295546
Epoch: 4
100%|███████████████████| 4672/4672 [06:43<00:00, 11.57it/s]
Training Accuracy: 67.08054589242708     Training Loss:0.7555616008650435
100%|███████████████████| 1001/1001 [00:53<00:00, 18.61it/s]
Validation Accuracy: 66.43356643356644   Validation Loss:0.7459081425652518
Epoch: 5
100%|███████████████████| 4672/4672 [06:47<00:00, 11.47it/s]
Training Accuracy: 69.03398447952904     Training Loss:0.7155961329109444
100%|███████████████████| 1001/1001 [00:54<00:00, 18.44it/s] 
Validation Accuracy: 70.8041958041958    Validation Loss:0.6976943148949465
Calculating test accuracy CNN model cnn1
Test ACCuracy: 71.2037962037962  Test Loss: 0.6891875450613794
------------------------------------------------------------------------------------------------------------------------------------------------------
______________________________________________________________________________________________________________________________________________________
Best test accuracy achieved by model cnn{index}:  71.2037962037962
CNN(
  (network): Sequential(
    (0): SkipLayer(
      (skiplayers): Sequential(
        (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (skip_connection): Conv2d(3, 32, kernel_size=(1, 1), stride=(1, 1))
      (relu): ReLU()
    )
    (1): Pooling(
      (pool): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    )
    (2): Pooling(
      (pool): AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (classifier): Sequential(
    (0): Linear(in_features=32, out_features=32, bias=True)
    (1): ReLU()
    (2): Linear(in_features=32, out_features=4, bias=True)
  )
)
best config [('conv', 32, 32), ('pool', 1, 32), ('pool', 0, 32)]
history {
    'cnn0': {
        'trainloss':
         [1.0508242789209399, 0.8835673358543354, 0.809855165998595, 0.7608422355639448, 0.7230342619339553],
          'trainacc':
           [52.49130318437249, 60.28900187316029, 63.60717152796361, 65.99411292480599, 68.32218356970833],
            'valloss':
             [0.9246639279457001, 0.8518805852630636, 0.7923010227444408, 0.7738665017810139, 0.8454483170899656],
              'valacc':
               [56.64335664335665, 59.96503496503497, 63.51148851148851, 64.83516483516483, 62.06293706293706]},
    'cnn1': {
        'trainloss':
         [1.0577738225932092, 0.8753257903688243, 0.8081053741330485, 0.7555616008650435, 0.7155961329109444],
          'trainacc':
           [52.66256355365266, 60.81883864062082, 64.23334225314423, 67.08054589242708, 69.03398447952904],
            'valloss':
             [0.9101097066919287, 0.8394484582629713, 0.7697237971295546, 0.7459081425652518, 0.6976943148949465], 
             'valacc':
              [58.56643356643357, 62.837162837162836, 66.4085914085914, 66.43356643356644, 70.8041958041958]
              }
              }
```
## 2. How to save the model
```python
model.save(path='./best_models/',filename='Model.pth') # these are the DEFAULT path and model name
# you may change the path or the model name according to your need
```
## 3. To check the summary of the model
```python
model.summary(input_shape=(3,32,32))
# this method will print the keras like summary of the model
```


Output Example:
```
               Layer    Output Shape                Kernal Shape        #params                 #(weights + bias)       requires_grad
------------------------------------------------------------------------------------------------------------------------------------------------------
            Conv2d-1    [1, 32, 32, 32]            [32, 3, 3, 3]        896                     (864 + 32)              True True
       BatchNorm2d-2    [1, 32, 32, 32]                 [32]            64                      (32 + 32)               True True
              ReLU-3    [1, 32, 32, 32]
            Conv2d-4    [1, 32, 32, 32]            [32, 32, 3, 3]       9248                    (9216 + 32)             True True
       BatchNorm2d-5    [1, 32, 32, 32]                 [32]            64                      (32 + 32)               True True
            Conv2d-6    [1, 32, 32, 32]            [32, 3, 1, 1]        128                     (96 + 32)               True True
              ReLU-7    [1, 32, 32, 32]
         MaxPool2d-8    [1, 32, 16, 16]
         AvgPool2d-9    [1, 32, 8, 8]
AdaptiveAvgPool2d-10    [1, 32, 1, 1]
           Linear-11    [1, 32]                       [32, 32]          1056                    (1024 + 32)             True True
             ReLU-12    [1, 32]
           Linear-13    [1, 4]                        [4, 32]           132                     (128 + 4)               True True
______________________________________________________________________________________________________________________________________________________

Total parameters 11,588
Total Non-Trainable parameters 0
Total Trainable parameters 11,588
(11588, 11588, 0)

```