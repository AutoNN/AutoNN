# How to use AutoNN GUI for Image Dataset


## The GUI interface

Steps to run GUI :

- Open terminal
- Run the following command 
```
pip install nocode-autonn 
autonn
```

![The GUI interface](/gui/screenshots/1.png)

## <span style="color :yellow">Buttons</span> 
 

-  <span style="color :violet">Open folder</span> : Used to select the path to the training dataset
-  <span style="color :violet">Show Configs</span> : Displays all the initial settings before training process
-  <span style="color :violet">Predict</span> : To make predictions on selected images
-  <span style="color :violet">Load Model</span> : To load a trained model 
-  <span style="color :violet">Display Graphs</span> : Displays the **_Training loss/accuracy vs Validation loss/accuracy_** of the generated models **only** after training when pressed
-  <span style="color :violet">Open Test Folder</span> : Used to select the path to the test dataset
-  <span style="color :violet">Start Training</span> : Starts the model training process when pressed
-  <span style="color :violet">Augment Dataset</span> : Augments the dataset when pressed
-  <span style="color :violet">Save Trained Model</span> : To save the model as `model_name.pth` file


## <span style="color :yellow">Radio Buttons</span> 
-  <span style="color :violet">Split required</span> : Select this button IF and only IF there is no separate `test dataset`
-  <span style="color :violet">Split NOT required</span> : `Default selection` | Select this if you have both the training and test dataset, provide the path to both datasets by clicking `Open folder` and `Open Test Folder` buttons

## <span style="color :yellow">Entry Text Boxes</span>

-  <span style="color :violet">Learning Rate</span> : `float` | Set the Learning rate for training
-  <span style="color :violet">Enter number of Channels</span> : `int` | Select the number of channels in the given training image 
-  <span style="color :violet">Enter number of Classes</span> : `int` | Enter the total number of classes to be classified in 
-  <span style="color :red">Enter image shape</span> : `str` | **height** x **width** | Should be a string and in the following format `32x32`


### Results upon pressing `Display graphs` after completion of the training process
![Alt text](/gui/screenshots/2.png)

### Saving the model after training
![Alt text](/gui/screenshots/3.png)

### Select the path where you want to store the trained model
![Alt text](/gui/screenshots/4.png)

### To load the trained model 

- Select `Load Model`

![Alt text](/gui/screenshots/5.png)