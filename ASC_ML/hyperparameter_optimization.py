from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow as tf
import numpy as np

from tensorflow.keras import backend as K

class Hyperparameter_Optimization:
    def __init__(self, X, Y, model, loss_fn, lr_opt = True, batch_opt = True, random_opt = True):
        self._train_X = X
        self._train_Y = Y
        self._model = model
        self._loss_fn = loss_fn
        self._lr_opt = lr_opt
        self._batch_opt = batch_opt
        self._random_opt = random_opt
    
    def _reinitialize_model(self, initializer = tf.keras.initializers.random_uniform()):
        for layer in self._model.layers:
            layer.set_weights([initializer(shape=w.shape) for w in layer.get_weights()])
    
    def get_best_lr(self, batch_size):
        # Get Range of Learning Rate vs loss funtion
        self._reinitialize_model()
        epoch = 50
        optimizer = Adam(lr = 1e-3)
        self._model.compile(loss = self._loss_fn, optimizer = optimizer)
        lr_schedule = LearningRateScheduler(lambda epoch : 1e-5 * 10**(epoch/10))
        history = self._model.fit(self._train_X, self._train_Y, epochs = epoch, batch_size = batch_size, callbacks = [lr_schedule])

        # Select best Learning Rate
        lrs = 1e-5 * 10**(np.arange(epoch)/10)
        losses = history.history["loss"].copy()
        min_loss = min(losses)
        best_lr = round(lrs[losses.index(min(losses))],5)

        # Re-Train using best Learning Rate
        # self._reinitialize_model()
        # epoch = 100
        # optimizer_new = Adam(lr = best_lr)
        # self._model.compile(loss = "mean_squared_error", optimizer = optimizer_new)
        # history = self._model.fit(self._train_X, self._train_Y, epochs = epoch, batch_size = 32)

        return best_lr, min_loss

    def set_batch_size(self):
        batch_size_list = [16,32,64]
        best_lr_batch_size = []
        for batch_size in batch_size_list:
            best_lr_list.append(self.get_best_lr(batch_size))
        return 
    
    def get_best_hyperparameters(self):



