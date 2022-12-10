import os
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.initializers import RandomUniform, GlorotUniform, GlorotNormal, HeUniform, HeNormal, LecunNormal, LecunUniform
from tensorflow.keras.optimizers import Adam

from AutoNN.networkbuilding import model_generation as model_gen
from AutoNN.networkbuilding import hyperparameter_optimization as hyp_opt
from AutoNN.networkbuilding.dropout_optimization import Dropout_Optimization
from AutoNN.networkbuilding.utilities import get_loss_function

class Model_Stacking:
    def __init__(self, train_x, train_y, test_x, test_y, loss_fn, model_path_list, model_conf_list, save_dir = ""):
        self._model_path_list = model_path_list
        self._model_conf_list = model_conf_list
        self._train_x = train_x
        self._train_y = train_y
        self._test_x = test_x
        self._test_y = test_y
        self._loss_fn = loss_fn
        self._save_dir = save_dir
        self._stacked_model_paths = []
        self._stacked_models = []

    def _stacked_model_generator(self):
        for path in self._model_path_list:
            for model_conf in self._model_conf_list:
                if self._model_path_list.index(path) == self._model_conf_list.index(model_conf):
                    continue
                model1 = load_model(path)
                activation = model1.layers[1].get_config()["activation"]
                reduced_model1 = Model(name = model1.name+"_reduced", inputs = model1.input, outputs = model1.layers[-3].output)
                last_layer = reduced_model1.output
                x = last_layer
                model2_obj = model_gen.NN_ModelGeneration(*model_conf)
                x = Dropout(0.0)(x)
                for layer_name in model2_obj.layer_conf:
                    x = Dense(model2_obj.layer_conf[layer_name], activation = activation, name = layer_name+"_2nd")(x)
                x = Dropout(0.0)(x)
                output_layer = Dense(model2_obj.output_layer_conf[0], activation = model2_obj.output_layer_conf[1], name = "output_layer" + "_" + model2_obj.model_name)(x)

                stacked_model = Model(name = model1.name + "_st_" + model2_obj.model_name,inputs = reduced_model1.input, outputs = output_layer)
                x = None
                output_layer = None
                yield stacked_model

    def optimize_stacked_models(self):
        stacked_model_generator = self._stacked_model_generator()
        for model in stacked_model_generator:
            print("-------------------------------------------------------------------------------------------------------------------")
            print(model.name)
            print(model.summary())
            h = hyp_opt.Hyperparameter_Optimization([self._train_x], [self._train_y], [self._test_x], [self._test_y], model, self._loss_fn, activation_opt = True, initializer_opt = True)
            best_lr, best_batch_size, best_activation, best_initializer, _ = h.get_best_hyperparameters()
            print(best_initializer)
            self._reinitialize_model(model, best_initializer)
            
            dr = Dropout_Optimization(self._train_x, self._train_y, self._test_x, self._test_y, self._loss_fn, epochs = 100, model = model)
            best_dropout_rates = dr.dropout_optimization(lr = best_lr, batch_size = best_batch_size, 
                                                                activation = best_activation, initializer = best_initializer, epoch = 100)
            print(f"DROPOUT RATES : {best_dropout_rates}")
            
            model = self._train_models(model = model, lr = best_lr, batch_size = best_batch_size)
            # self._save_model(model)
            self._stacked_models.append(model)
    
    def _train_models(self, model, lr, batch_size):
        optimizer = Adam(learning_rate = lr)
        model.compile(loss = self._loss_fn, optimizer = optimizer)
        history = model.fit(self._train_x, self._train_y, epochs = 100, batch_size = batch_size, verbose = 0)

        scores = model.evaluate(self._train_x, self._train_y, verbose = 0)
        scores_test = model.evaluate(self._test_x, self._test_y, verbose = 0)
        print(f"TRAIN_LOSS = {scores}, TEST_LOSS = {scores_test}")
        return model
        
    @staticmethod
    def _reinitialize_model(model, initializer_str = "RandomUniform"):
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
        for layer in model.layers:
            layer.set_weights([initializer(shape=w.shape) for w in layer.get_weights()])
            
    def save_model(self):
        for model in self._stacked_models:
            path = os.path.join(self._save_dir,model.name)
            model.save(path)
            self._stacked_model_paths.append(path)
