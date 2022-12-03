# Image Classification Using AutoNN


AutoNN's CNN (Convolutional Neural Network) Models are built on PyTorch.




## 1. How to train a CNN model for image classification:

Import `CreateCNN` from CNN's `cnn_generator` module

Example
```python
from AutoNN.CNN.cnn_generator import CreateCNN, CNN 

inst = CreateCNN()

model,model_config,history=inst.get_bestCNN(path_trainset="E:/output/cifar10/cifar10/train",
path_testset="E:/output/cifar10/cifar10/test",
split_required=False,EPOCHS=10,image_shape=(32,32))
```

??? OUTPUT

    ```
      ░█▀▀█ █░░█ ▀▀█▀▀ █▀▀█ ▒█▄░▒█ ▒█▄░▒█ 
      ▒█▄▄█ █░░█ ░░█░░ █░░█ ▒█▒█▒█ ▒█▒█▒█ 
      ▒█░▒█ ░▀▀▀ ░░▀░░ ▀▀▀▀ ▒█░░▀█ ▒█░░▀█

      Version: 2.0.0
      An AutoML framework by
      Anish Konar, Arjun Ghosh, Rajarshi Banerjee, Sagnik Nayak.

      Default computing platform: cuda
      Classes:  ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] # Classes:  10
      Training set size: 40000 | Validation Set size: 10000 | Test Set size: 10000
      Architecture search Complete..! Time Taken:  0:00:01.688582
      Number of models generated: 2
      Searching for the best model. Please be patient. Thank you....
      Training CNN model cnn0
      Epoch: 1/10
      100%|██████████| 2500/2500 [01:08<00:00, 36.24it/s]
      Training Accuracy: 35.1775	 Training Loss:1.7207594780921935
      100%|██████████| 625/625 [00:09<00:00, 65.11it/s]
      Validation Accuracy: 43.38	 Validation Loss:1.5083902006149292
      Epoch: 2/10
      100%|██████████| 2500/2500 [01:03<00:00, 39.34it/s]
      Training Accuracy: 44.7975	 Training Loss:1.483097927236557
      100%|██████████| 625/625 [00:08<00:00, 70.63it/s]
      Validation Accuracy: 47.59	 Validation Loss:1.4107291974067688
      Epoch: 3/10
      100%|██████████| 2500/2500 [01:03<00:00, 39.67it/s]
      Training Accuracy: 49.26	 Training Loss:1.3825817276716232
      100%|██████████| 625/625 [00:08<00:00, 70.44it/s]
      Validation Accuracy: 49.26	 Validation Loss:1.3538662633895875
      Epoch: 4/10
      100%|██████████| 2500/2500 [01:02<00:00, 39.78it/s]
      Training Accuracy: 51.4975	 Training Loss:1.3264493201732634
      100%|██████████| 625/625 [00:08<00:00, 70.79it/s]
      Validation Accuracy: 54.59	 Validation Loss:1.263453982925415
      Epoch: 5/10
      100%|██████████| 2500/2500 [01:03<00:00, 39.61it/s]
      Training Accuracy: 53.3775	 Training Loss:1.2788944167137146
      100%|██████████| 625/625 [00:08<00:00, 70.93it/s]
      Validation Accuracy: 53.4	 Validation Loss:1.2734063954353332
      Epoch: 6/10
      100%|██████████| 2500/2500 [01:03<00:00, 39.67it/s]
      Training Accuracy: 54.59	 Training Loss:1.2445036834478378
      100%|██████████| 625/625 [00:08<00:00, 70.81it/s]
      Validation Accuracy: 54.01	 Validation Loss:1.2412574873924256
      Epoch: 7/10
      100%|██████████| 2500/2500 [01:03<00:00, 39.66it/s]
      Training Accuracy: 56.045	 Training Loss:1.2121670446991921
      100%|██████████| 625/625 [00:08<00:00, 70.66it/s]
      Validation Accuracy: 57.49	 Validation Loss:1.1828145512580872
      Epoch: 8/10
      100%|██████████| 2500/2500 [01:02<00:00, 39.71it/s]
      Training Accuracy: 56.75	 Training Loss:1.1846893740534783
      100%|██████████| 625/625 [00:08<00:00, 70.39it/s]
      Validation Accuracy: 58.49	 Validation Loss:1.1632665138721465
      Epoch: 9/10
      100%|██████████| 2500/2500 [01:02<00:00, 39.73it/s]
      Training Accuracy: 58.13	 Training Loss:1.153260654783249
      100%|██████████| 625/625 [00:08<00:00, 70.20it/s]
      Validation Accuracy: 60.65	 Validation Loss:1.1054361756324769
      Epoch: 10/10
      100%|██████████| 2500/2500 [01:02<00:00, 40.28it/s]
      Training Accuracy: 58.895	 Training Loss:1.1318354423761368
      100%|██████████| 625/625 [00:08<00:00, 73.48it/s]
      Validation Accuracy: 59.0	 Validation Loss:1.1538037222385407
      Calculating test accuracy CNN model cnn0
      Test ACCuracy: 59.84	 Test Loss: 1.13254158577919
      ------------------------------------------------------------------------------------------------------------------------------------------------------
      ______________________________________________________________________________________________________________________________________________________
      Training CNN model cnn1
      Epoch: 1/10
      100%|██████████| 2500/2500 [00:41<00:00, 59.80it/s]
      Training Accuracy: 30.5625	 Training Loss:1.8546679210186006
      100%|██████████| 625/625 [00:08<00:00, 76.54it/s]
      Validation Accuracy: 33.79	 Validation Loss:1.7818523860931397
      Epoch: 2/10
      100%|██████████| 2500/2500 [00:41<00:00, 60.42it/s]
      Training Accuracy: 39.33	 Training Loss:1.6098018644332885
      100%|██████████| 625/625 [00:08<00:00, 76.10it/s]
      Validation Accuracy: 43.24	 Validation Loss:1.520786734867096
      Epoch: 3/10
      100%|██████████| 2500/2500 [00:41<00:00, 60.35it/s]
      Training Accuracy: 44.2325	 Training Loss:1.5038440037250518
      100%|██████████| 625/625 [00:08<00:00, 76.69it/s]
      Validation Accuracy: 45.07	 Validation Loss:1.4924028639793396
      Epoch: 4/10
      100%|██████████| 2500/2500 [00:41<00:00, 60.33it/s]
      Training Accuracy: 47.06	 Training Loss:1.438860204720497
      100%|██████████| 625/625 [00:08<00:00, 76.10it/s]
      Validation Accuracy: 47.9	 Validation Loss:1.4355408569335937
      Epoch: 5/10
      100%|██████████| 2500/2500 [00:41<00:00, 60.41it/s]
      Training Accuracy: 49.3625	 Training Loss:1.3855291500329971
      100%|██████████| 625/625 [00:08<00:00, 76.00it/s]
      Validation Accuracy: 51.42	 Validation Loss:1.328598715877533
      Epoch: 6/10
      100%|██████████| 2500/2500 [00:41<00:00, 59.96it/s]
      Training Accuracy: 51.2	 Training Loss:1.3459105017662047
      100%|██████████| 625/625 [00:09<00:00, 66.04it/s]
      Validation Accuracy: 53.64	 Validation Loss:1.2791677060127258
      Epoch: 7/10
      100%|██████████| 2500/2500 [00:44<00:00, 56.65it/s]
      Training Accuracy: 52.4025	 Training Loss:1.3105635814905168
      100%|██████████| 625/625 [00:08<00:00, 69.72it/s]
      Validation Accuracy: 52.93	 Validation Loss:1.2803085575103759
      Epoch: 8/10
      100%|██████████| 2500/2500 [00:44<00:00, 56.75it/s]
      Training Accuracy: 53.41	 Training Loss:1.2841510282278061
      100%|██████████| 625/625 [00:09<00:00, 69.40it/s]
      Validation Accuracy: 55.3	 Validation Loss:1.2356650713443755
      Epoch: 9/10
      100%|██████████| 2500/2500 [00:43<00:00, 57.20it/s]
      Training Accuracy: 54.435	 Training Loss:1.2552653106451035
      100%|██████████| 625/625 [00:09<00:00, 67.85it/s]
      Validation Accuracy: 57.19	 Validation Loss:1.1992870372772217
      Epoch: 10/10
      100%|██████████| 2500/2500 [00:43<00:00, 57.08it/s]
      Training Accuracy: 55.0625	 Training Loss:1.2330288346767426
      100%|██████████| 625/625 [00:09<00:00, 67.08it/s]
      Validation Accuracy: 56.51	 Validation Loss:1.2072079391002655
      Calculating test accuracy CNN model cnn1
      Test ACCuracy: 55.25	 Test Loss: 1.2267251291751862
      ------------------------------------------------------------------------------------------------------------------------------------------------------
      ______________________________________________________________________________________________________________________________________________________
      Best test accuracy achieved by model cnn0:  59.84

    ```

### Print the model:
```python
print(model)
```

??? OUTPUT
    ```
    CNN(
      (network): Sequential(
        (0): SkipLayer(
          (skiplayers): Sequential(
            (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (skip_connection): Conv2d(3, 16, kernel_size=(1, 1), stride=(1, 1))
          (relu): ReLU()
        )
        (1): SkipLayer(
          (skiplayers): Sequential(
            (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(16, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (skip_connection): Conv2d(16, 128, kernel_size=(1, 1), stride=(1, 1))
          (relu): ReLU()
        )
      )
      (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
      (classifier): Sequential(
        (0): Linear(in_features=128, out_features=32, bias=True)
        (1): ReLU()
        (2): Linear(in_features=32, out_features=10, bias=True)
      )
    )
    ```


### Get list of all classes. Example:
```python
print(inst.get_classes)
```

??? OUTPUT
    ```
    ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    ```

    



### Print Model configuration:

```python
print(model_config)
```
??? OUTPUT
    ```
    [('conv', 16, 16), ('conv', 16, 128)]
    ```
### Print history of all training data:


```python
print(history)
```
??? Output
    ```{r}
    {
      'cnn0': {
        'trainloss': [1.7207594780921935,   1.483097927236557,   1.3825817276716232,   1.3264493201732634,   1.2788944167137146,   1.2445036834478378,   1.2121670446991921,   1.1846893740534783,   1.153260654783249,
      1.1318354423761368],
      'trainacc': [35.1775,   44.7975,   49.26,   51.4975,   53.3775,   54.59,   56.045,   56.75,   58.13,   58.895],
      'valloss': [1.5083902006149292,   1.4107291974067688,   1.3538662633895875,   1.263453982925415,   1.2734063954353332,   1.2412574873924256,   1.1828145512580872,   1.1632665138721465,   1.1054361756324769,   1.1538037222385407],
      'valacc': [43.38,   47.59,   49.26,   54.59,   53.4,   54.01,   57.49,   58.49,   60.65,   59.0]},
    
    'cnn1': {
      'trainloss': [1.8546679210186006,   1.6098018644332885,   1.5038440037250518,   1.438860204720497,   1.3855291500329971,   1.3459105017662047,   1.3105635814905168,   1.2841510282278061,   1.2552653106451035,   1.2330288346767426],
      'trainacc': [30.5625,   39.33,   44.2325,   47.06,   49.3625,   51.2,   52.4025,   53.41,   54.435,   55.0625],
      'valloss': [1.7818523860931397,   1.520786734867096,   1.4924028639793396,   1.4355408569335937,   1.328598715877533,   1.2791677060127258,   1.2803085575103759,   1.2356650713443755,   1.1992870372772217,   1.2072079391002655],
      'valacc': [33.79,   43.24,   45.07,   47.9,   51.42,   53.64,   52.93,   55.3,   57.19,   56.51]
      }
    }
    ```




## 2. Plot the Training loss vs Validation loss:

```python
from AutoNN.CNN.utils.EDA import plot_graph

plot_graph(history)
```


## 3. How to save the model:
```python
model.save(inst.get_classes,inst.get_imageshape,path='./best models/',filename='mnistmodel.pth',
    config_file_path='./best models/',config_filename = 'mnistcfg.json')
```

## 4. To check the summary of the model:
```python
model.summary(input_shape=(3,32,32))
# this method will print the keras like summary of the model
```



## 5. How to load saved model:
```python
myModel = CNN(3,10)
myModel.load(PATH='./best models/mnistmodel.pth',config_path="./best models/mnistcfg.json",printmodel=True)
```
??? OUTPUT
    ```{r}
      Network Architecture loaded!
      CNN(
        (network): Sequential(
          (0): SkipLayer(
            (skiplayers): Sequential(
              (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU()
              (3): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (skip_connection): Conv2d(3, 16, kernel_size=(1, 1), stride=(1, 1))
            (relu): ReLU()
          )
          (1): SkipLayer(
            (skiplayers): Sequential(
              (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU()
              (3): Conv2d(16, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (skip_connection): Conv2d(16, 128, kernel_size=(1, 1), stride=(1, 1))
            (relu): ReLU()
          )
        )
        (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
        (classifier): Sequential(
          (0): Linear(in_features=128, out_features=32, bias=True)
          (1): ReLU()
          (2): Linear(in_features=32, out_features=10, bias=True)
        )
      )
      Loading complete, your model is now ready for evaluation!
    ``` 


## 6. How to use Loaded model on new images:

```python
test_images = ['E:/output/cifar10/cifar10/test/bird/0012.png', # bird
'E:/output/cifar10/cifar10/train/ship/0002.png'] # ship

myModel.predict(paths=test_images)

```
??? OUTPUT
    ```
    ['bird', 'ship']
    ```
