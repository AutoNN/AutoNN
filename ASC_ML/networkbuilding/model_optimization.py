from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import RandomUniform, GlorotUniform, GlorotNormal, HeUniform, HeNormal, LecunUniform, LecunNormal
from tensorflow.keras.activations import tanh, relu, selu
from ASC_ML.networkbuilding import hyperparameter_optimization as hyp_opt
from ASC_ML.networkbuilding import model_generation as model_gen
from ASC_ML.networkbuilding.utilities import get_loss_function

import os

class Model_Optimization:

    def __init__(self, train_x, train_y, test_x, test_y, epochs, model_dict_list, save_dir):
        """Creates all parallel models using base models from model_generation, trains and evaluates each model, stores best models.
        Parameters
        ----------
        train_x : numpy array
            Final processed Training Dataset in numpy array format
        train_y : numpy array
            Final processed Labels
        epochs : Number of epochs run per parallel model
            batch_size : Batch Size of training data for training
        Returns
        -------
        
        """
        self._train_x = train_x
        self._train_y = train_y
        self._test_x = test_x
        self._test_y = test_y
        self._epochs = epochs
        self._model_dict_list = model_dict_list
        self._save_dir = save_dir
        self._saved_paths = []
        self._model_confs = []
        self._best_hyp_permodel = []

    @property
    def saved_paths(self):
        return self._saved_paths
    
    @property
    def model_confs(self):
        return self._model_confs

    @property
    def best_hyp_permodel(self):
        return self._best_hyp_permodel

    def get_loss_function(self):
        # Logic to get loss funtion
        # return "mean_absolute_percentage_error"
        # return self.root_mean_squared_error
        # return "mean_squared_error"
        return "mean_absolute_error"
    
    def train_models_2(self):
        for model_dict in self._model_dict_list:
            model = load_model(model_dict["path_weights"])
            optimizer = Adam(lr = 1e-3)
            model.compile(loss = "mean_squared_error", optimizer = optimizer)
            history = model.fit(self._train_x, self._train_y, epochs = 200, batch_size = 64)
    
    def _candidate_model_generator(self):
        for model_dict in self._model_dict_list:
            model_conf = model_dict["model_conf"]
            input_layer_list, output_layer_list = self._get_input_output_layer_list([model_conf])
            model = Model(name = model_dict["model_name"],inputs = input_layer_list, outputs = output_layer_list)
            # model = load_model(model_dict["path_weights"])
            self._model_confs.append(model_conf)
            yield model
    
    def optimize_models(self, save):
        # loss_fn = self.get_loss_function()
        loss_fn = get_loss_function()
        candidate_model_generator = self._candidate_model_generator()
        for model in candidate_model_generator:
            # print(model.summary())
            print("--------------------------------------------------------------------------------")
            print(f"Model Name : {model.name}\n")
            h = hyp_opt.Hyperparameter_Optimization([self._train_x], [self._train_y], model, loss_fn)
            best_lr, best_batch_size, best_activation, best_initializer = h.get_best_hyperparameters()
            self._reinitialize_model(model, best_initializer)
            self._set_activation(model, best_activation)
            self._best_hyp_permodel.append([best_lr, best_batch_size, best_activation, best_initializer])
            model,_,_,_ = self.train_model(input_data = [self._train_x, self._train_y, self._test_x, self._test_y], Pmodel=model, n_model=1,
                                epochs=200,loss_fn=loss_fn, lr=best_lr, batch_size=best_batch_size)
            if save==True:
                self.save_weights(model)
    
    def save_weights(self, model):
        # save_dir = os.path.join(self._save_dir,"/candidate_models")
        save_path = os.path.join(self._save_dir, model.name)
        # if os.path.isdir(save_dir):
        #     os.mkdir(save_dir)
        model.save(save_path)
        self._saved_paths.append(save_path)
                
    def train_model(self, input_data, Pmodel, n_model, epochs, loss_fn, lr = 1e-3, batch_size = 64, activation = None, initializer = None):

        # if activation != None:
        #     Pmodel = self._set_activation(Pmodel, activation)
        # if initializer != None:    
        #     Pmodel = self._reinitialize_model(Pmodel, initializer)

        input_x, input_labels, input_test_x, input_test_labels = input_data
        optimizer = Adam(lr = lr)
        Pmodel.compile(loss = loss_fn, optimizer = optimizer)
        history = Pmodel.fit(input_x, input_labels, epochs = epochs, batch_size = batch_size, verbose = 0)

        metrics_names, scores, scores_test = self.print_scores(Pmodel, input_data)
        
        return Pmodel, metrics_names, scores, scores_test

    def print_scores(self, Pmodel, input_data):
        input_x, input_labels, input_test_x, input_test_labels = input_data
        scores = Pmodel.evaluate(input_x, input_labels, verbose = 0)
        scores_test = Pmodel.evaluate(input_test_x, input_test_labels, verbose = 0)

        print("\n \n \n")

        metrics_names = Pmodel.metrics_names
        metrics_names = [metrics_names] if not isinstance(metrics_names, list) else metrics_names
        scores = [scores] if not isinstance(scores, list) else scores
        scores_test = [scores_test] if not isinstance(scores_test, list) else scores_test

        for name,score,score_test in zip(metrics_names, scores, scores_test):
            print(name, " : ", score, ", TEST : ", score_test)
        return metrics_names, scores, scores_test

    @staticmethod
    def _get_input_output_layer_list(model_list):
        input_layer_list = []
        output_layer_list = []
        for nn_model_conf in model_list:
            nn_model = model_gen.NN_ModelGeneration(*nn_model_conf)
            input_layer_list.append(nn_model.input_layer)
            output_layer_list.append(nn_model.output_layer)
        return input_layer_list, output_layer_list

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

    @staticmethod
    def _set_activation(model, activation_str = "relu"):
        # activation = activation string
        if activation_str == "relu" : activation = relu
        elif activation_str == "tanh" : activation = tanh
        elif activation_str == "selu" : activation = selu
        for layer in model.layers:
            layer.activation = activation