from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.activations import tanh, relu, selu
from tensorflow.keras.initializers import RandomUniform, GlorotUniform, GlorotNormal, HeUniform, HeNormal, LecunNormal, LecunUniform
from numpy.random import seed
# tensorflow.random.set_seed
import tensorflow as tf
import numpy as np

from tensorflow.keras import backend as K

class Hyperparameter_Optimization:
    def __init__(self, X, Y, test_X, test_Y, model, loss_fn, lr_opt = True, batch_opt = True, activation_opt = True, initializer_opt = True, dropout_opt = False):
        self._seed = 420
        self._train_X = X
        self._train_Y = Y
        self._test_X = test_X
        self._test_Y = test_Y
        self._model = model
        self._loss_fn = loss_fn
        self._lr_opt = lr_opt
        self._batch_opt = batch_opt
        self._activation_opt = activation_opt
        self._initializer_opt = initializer_opt
        self._dropout_opt = dropout_opt
        self._current_activation = None
        self._hyperparameter_list = [] # [best_batch_size, best_lr, best_activation_function, best_initializer, best_seed]

    @property
    def current_activation(self):
        self._current_activation = self._model.layers[-2].get_config()["activation"]
        return self._current_activation
    
    def _set_activation(self, activation_str):
        # activation = activation string
        if activation_str == "relu" : activation = relu
        elif activation_str == "tanh" : activation = tanh
        elif activation_str == "selu" : activation = selu
        for layer in self._model.layers[1:-1]:
            layer.activation = activation
    
    def _reinitialize_model(self, initializer_str = "RandomUniform"):
        if initializer_str == "RandomUniform":
            initializer = RandomUniform(seed = 420)
        elif initializer_str == "GlorotUniform":
            initializer = GlorotUniform(seed = 420)
        elif initializer_str == "HeUniform":
            initializer = HeUniform(seed = 420)
        elif initializer_str == "GlorotNormal":
            initializer = GlorotNormal(seed = 420)
        elif initializer_str == "HeNormal":
            initializer = HeNormal(seed = 420)
        elif initializer_str == "LecunNormal":
            initializer = LecunNormal(seed = 420)
        elif initializer_str == "LecunUniform":
            initializer = LecunUniform(seed = 420)
        for layer in self._model.layers:
            layer.set_weights([initializer(shape=w.shape) for w in layer.get_weights()])
    
    def get_best_lr(self, batch_size):
        # Get Range of Learning Rate vs loss funtion
        # self._reinitialize_model()
        epoch = 50
        optimizer = Adam(learning_rate = 1e-3)
        self._model.compile(loss = self._loss_fn, optimizer = optimizer)
        lr_schedule = LearningRateScheduler(lambda epoch : 1e-5 * 10**(epoch/10))
        history = self._model.fit(self._train_X, self._train_Y, epochs = epoch, batch_size = batch_size, callbacks = [lr_schedule], verbose = 0)

        # Select best Learning Rate
        lrs = 1e-5 * 10**(np.arange(epoch)/10)
        losses = history.history["loss"].copy()
        min_loss = min(losses)
        best_lr = round(lrs[losses.index(min(losses))],5)

        return best_lr, min_loss
    
    def _train_model(self, lr, batch_size):
        optimizer = Adam(learning_rate = lr)
        self._model.compile(loss = self._loss_fn, optimizer = optimizer)
        history = self._model.fit(self._train_X, self._train_Y, epochs = 25, batch_size = batch_size, verbose = 0)
        score = self._model.evaluate(self._train_X, self._train_Y, verbose = 0)
        return score


    def get_best_batch_size_lr(self, initializer_str):
        batch_size_list = [16,32,64,128]
        best_lr_list = []
        best_loss_list = []
        for batch_size in batch_size_list:
            # print(f"\nSETTING BATCH SIZE TO {batch_size}")
            self._reinitialize_model(initializer_str=initializer_str)
            best_lr_batch, _ = self.get_best_lr(batch_size)

            self._reinitialize_model(initializer_str=initializer_str)
            best_loss_batch = self._train_model(best_lr_batch, batch_size)
            
            best_lr_list.append(best_lr_batch)
            best_loss_list.append(best_loss_batch)

        best_lr = best_lr_list[best_loss_list.index(min(best_loss_list))]
        best_batch_size = batch_size_list[best_loss_list.index(min(best_loss_list))]
        min_loss = min(best_loss_list)

        # print(f"BEST LR = {best_lr}    BEST BATCH SIZE = {best_batch_size}")
        return best_lr, best_batch_size, min_loss
    
    def get_best_hyperparameters(self):

        best_loss = 0
        best_lr = None
        best_activation = None
        best_batch_size = None
        best_initializer = None
        best_dropout_rate = None
        best_dropout_loss = None
        # activation_list = ["relu","tanh","selu"]
        activation_list = ["relu","selu","tanh"]
        # intializer_list = [["GlorotUniform","GlorotNormal"],["HeUniform","HeNormal"],["GlorotUniform","GlorotNormal"]]
        # intializer_list = [["GlorotUniform","GlorotNormal"],["LecunUniform","LecunNormal"]]
        intializer_list = [["GlorotUniform", "LecunUniform", "HeUniform"], ["GlorotUniform", "LecunUniform", "HeUniform"], ["GlorotUniform", "LecunUniform", "HeUniform"], ["GlorotUniform", "LecunUniform", "HeUniform"]]
        i = 0
        if self._activation_opt == True:
            for activation in activation_list:
                self._set_activation(activation)
                if self._initializer_opt == True:
                    for initializer in intializer_list[i]:
                        self._reinitialize_model(initializer_str=initializer)
                        lr, batch_size, loss = self.lr_batch_optimization(initializer_str=initializer)
                        if(best_loss == 0 or loss<best_loss):
                            best_loss = loss
                            best_lr = lr
                            best_batch_size = batch_size
                            best_activation = activation
                            best_initializer = initializer
                else:
                    lr, batch_size, loss = self.lr_batch_optimization()
                    if(best_loss == 0 or loss<best_loss): 
                            best_loss = loss
                            best_lr = lr
                            best_batch_size = batch_size
                            best_activation = activation
                i = i+1
        else:
            best_lr, best_batch_size, best_loss = self.lr_batch_optimization()

        if self._dropout_opt == True:
            best_dropout_rate, best_dropout_loss = self.get_best_dropout(best_lr, best_batch_size, best_activation, best_initializer)
            
        print(f"BEST HYPERPARAMETERS : BEST_LOSS : {best_loss}, BEST_ACTIVATION : {best_activation}, BEST_INITIALIZER : {best_initializer}, BEST_LEARINING_RATE : {best_lr}, BEST_BATCHSIZE : {best_batch_size}, BEST_DROPOUT_RATE : {best_dropout_rate}, BEST_DROPOUT_LOSS : {best_dropout_loss}")
        return best_lr, best_batch_size, best_activation, best_initializer, best_dropout_rate
        
    def lr_batch_optimization(self, initializer_str = "RandomUniform"):
        if self._lr_opt == True and self._batch_opt == False:
            best_lr,best_loss = self.get_best_lr(batch_size=32)
            best_batch_size = 32
        elif self._batch_opt == True and self._lr_opt == True:
            best_lr, best_batch_size, best_loss = self.get_best_batch_size_lr(initializer_str = initializer_str)

        return best_lr, best_batch_size, best_loss

    def get_best_dropout(self, best_lr, best_batch_size, best_activation, best_initializer):
        dropout_list = [0.0, 0.2, 0.5]
        dropout_loss_list = []
        for dropout_rate in dropout_list:
            self._model.layers[-2].rate = dropout_rate

            if best_initializer == None:
                self._reinitialize_model()
            else: self._reinitialize_model(best_initializer)
            if best_activation != None:
                self._set_activation(best_activation)

            optimizer = Adam(learning_rate = best_lr)
            self._model.compile(loss = self._loss_fn, optimizer = optimizer)
            history = self._model.fit(self._train_X, self._train_Y, epochs = 50, batch_size = best_batch_size, verbose = 0)
            scores = self._model.evaluate(self._train_X, self._train_Y, verbose = 0)
            scores_test = self._model.evaluate(self._test_X, self._test_Y, verbose = 0)
            # dropout_loss_list.append(scores)
            dropout_loss_list.append(scores_test-scores)

        best_dropout_rate = dropout_list[dropout_loss_list.index(min(dropout_loss_list))]
        best_dropout_loss = min(dropout_loss_list)
        self._model.layers[-2].rate = best_dropout_rate

        return best_dropout_rate, best_dropout_loss
