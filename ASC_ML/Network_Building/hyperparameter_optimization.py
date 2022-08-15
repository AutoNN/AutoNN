from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.activations import tanh, relu
from tensorflow.keras.initializers import RandomUniform, GlorotUniform, GlorotNormal, HeUniform, HeNormal
from numpy.random import seed
from tensorflow import set_random_seed
import tensorflow as tf
import numpy as np

from tensorflow.keras import backend as K

class Hyperparameter_Optimization:
    def __init__(self, X, Y, model, loss_fn, lr_opt = True, batch_opt = True, activation_opt = True, initializer_opt = True, random_opt = True):
        self._seed = 420
        self._train_X = X
        self._train_Y = Y
        self._model = model
        self._loss_fn = loss_fn
        self._lr_opt = lr_opt
        self._batch_opt = batch_opt
        self._activation_opt = activation_opt
        self._initializer_opt = initializer_opt
        self._random_opt = random_opt
        self._current_activation = None
        self._hyperparameter_list = [] # [best_batch_size, best_lr, best_activation_function, best_initializer, best_seed]

    @property
    def current_activation(self):
        self._current_activation = self._model.layers[-2].get_config()["activation"]
        return self._current_activation
    
    def _set_activation(self, activation):
        # activation = activation function not string
        for layer in self._model.layers:
            layer.activation = activation
    
    def _reinitialize_model(self, initializer = RandomUniform(seed = 420)):
        for layer in self._model.layers:
            layer.set_weights([initializer(shape=w.shape) for w in layer.get_weights()])
    
    def get_best_lr(self, batch_size):
        # Get Range of Learning Rate vs loss funtion
        self._reinitialize_model()
        epoch = 50
        optimizer = Adam(lr = 1e-3)
        self._model.compile(loss = self._loss_fn, optimizer = optimizer)
        lr_schedule = LearningRateScheduler(lambda epoch : 1e-5 * 10**(epoch/10))
        history = self._model.fit(self._train_X, self._train_Y, epochs = epoch, batch_size = batch_size, callbacks = [lr_schedule], verbose = 0)

        # Select best Learning Rate
        lrs = 1e-5 * 10**(np.arange(epoch)/10)
        losses = history.history["loss"].copy()
        min_loss = min(losses)
        best_lr = round(lrs[losses.index(min(losses))],5)

        return best_lr, min_loss

    def get_best_batch_size_lr(self):
        print("----------------------------------------------------------------------------------------------------------------")
        batch_size_list = [16,32,64,128]
        best_lr_list = []
        best_loss_list = []
        for batch_size in batch_size_list:
            print(f"\nSETTING BATCH SIZE TO {batch_size}")
            best_lr_batch, best_loss_batch = self.get_best_lr(batch_size)
            best_lr_list.append(best_lr_batch)
            best_loss_list.append(best_loss_batch)

        best_lr = best_lr_list[best_loss_list.index(min(best_loss_list))]
        best_batch_size = batch_size_list[best_loss_list.index(min(best_loss_list))]
        min_loss = min(best_loss_list)

        print(f"BEST LR = {best_lr}    BEST BATCH SIZE = {best_batch_size}")
        return best_lr, best_batch_size, min_loss
    
    def get_best_hyperparameters(self):

        activation_list = [relu,tanh]
        intializer_list = [[GlorotUniform,GlorotNormal],[HeUniform,HeNormal]]

        if self._activation_opt == True:
            for activation in activation_list:
                self._set_activation(activation)
                if self._initializer_opt == True:
                    
                self._reinitialize_model(initializer=)
                if self._lr_opt == True and self._batch_opt == False:
                    best_lr,best_loss = self.get_best_lr(batch_size=32)
                    best_batch_size = None
                elif self._batch_opt == True and self._lr_opt == True:
                    best_lr, best_batch_size, best_loss = self.get_best_batch_size_lr()
                best_relu_loss = best_loss
                self._set_activation(tanh)
        



