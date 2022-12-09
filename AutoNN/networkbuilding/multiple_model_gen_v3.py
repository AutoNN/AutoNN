from AutoNN.networkbuilding import model_generation as model_gen
from AutoNN.networkbuilding import search_space_gen_v1 as search
from AutoNN.networkbuilding import hyperparameter_optimization as hyp_opt

from tensorflow.keras.activations import tanh, relu, selu
from tensorflow.keras.initializers import RandomUniform, GlorotUniform, GlorotNormal, HeUniform, HeNormal
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow as tf
import numpy as np

from tensorflow.keras import backend as K

class Multiple_Model_Gen_V3:

    def __init__(self, train_x, train_y, test_x, test_y, loss_fn, epochs, batch_size, input_shape, output_shape = 1, output_activation = None, max_no_layers = 3, model_per_batch = 10, save_dir = ""):
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
        self._loss_fn = loss_fn
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._epochs = epochs
        self._batch_size = batch_size
        self._output_activation = output_activation
        self._max_no_layers = max_no_layers
        self._model_per_batch = model_per_batch
        self._save_dir = save_dir
        self._model_confs = []
        self._evaluate_dict_list = []
        self._no_top_model = 2
        self.get_model_confs()
    
    @property
    def model_confs(self):
        # Return Model Name
        return self._model_confs

    @property
    def evaluate_dict_list(self):
        # Return Model Name
        return self._evaluate_dict_list

    def get_best_models(self, save = False):
        parallel_model_generator = self._parallel_model_generator()

        # 1st Loop, Get best model architectures
        for parallelModel, n, model_conf_batch_list in parallel_model_generator:
            input_data = self._get_train_lists(n)
            Pmodel, metrics_names, scores, scores_test = self.train_model(input_data = input_data, Pmodel = parallelModel, 
                                                                            epochs = self._epochs, n_model = n, loss_fn = self._loss_fn)
            self._evaluate_save_model(input_data = input_data, parallelModel = Pmodel, metrics_names=metrics_names, scores=scores_test, n=n, model_conf_batch_list = model_conf_batch_list)
        
        # 2nd Loop, Tune best model architectures
        # print(self._evaluate_dict_list)
        if save:
            self.save_weights()
        tf.keras.backend.clear_session()

    def train_model(self, input_data, Pmodel, n_model, epochs, loss_fn, lr = 1e-3, batch_size = 64, activation = None, initializer = None):
        input_x, input_labels, input_test_x, input_test_labels = input_data
        optimizer = Adam(learning_rate = lr)
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
    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

    def _evaluate_save_model(self, input_data, parallelModel, metrics_names, scores, n, model_conf_batch_list):
        # n : Number of models in the parallelModel
        # List of dictionaries {"model_name":"densexyz", "score":0.000, "path_weights":"/home/something/dense"}
        input_x, input_labels, _, _ = input_data
        sep_model_list = []
        for i in range(n):
            sep_model_list.append(Model(inputs = parallelModel.inputs[i], outputs = parallelModel.outputs[i])) 

        metrics_names = metrics_names[1:1+n]
        model_scores = scores[1:1+n]
        entry_flag = False

        for metric_name, model_score, model, model_conf in zip(metrics_names,model_scores,sep_model_list,model_conf_batch_list):
            model_name = metric_name.removeprefix("output_layer_")
            model_name = model_name.removesuffix("_loss")
            curr_model_dict = {"model_name":model_name, "score":model_score, "path_weights":self._save_dir + model_name, "model_conf":model_conf, "model":model}

            if len(self._evaluate_dict_list) == 0:
                self._evaluate_dict_list.append(curr_model_dict)
            else:
                index = 0
                for model_dict in self._evaluate_dict_list:
                    if(curr_model_dict["score"] < model_dict["score"]):
                        self._evaluate_dict_list.insert(index, curr_model_dict)  
                        entry_flag = True                      
                        if(len(self._evaluate_dict_list) > self._no_top_model): self._evaluate_dict_list = self._evaluate_dict_list[:self._no_top_model]
                        break
                    index = index + 1
                if(not entry_flag and len(self._evaluate_dict_list) < self._no_top_model):
                    self._evaluate_dict_list.append(curr_model_dict)
                entry_flag = False

    def save_weights(self):
        for dict in self._evaluate_dict_list:
            model = dict["model"]
            model.save(dict["path_weights"])

    def _parallel_model_generator(self):
        for batch in self._model_confs:
            input_layer_list, output_layer_list = self._get_input_output_layer_list(batch)
            n = len(input_layer_list)
            parallelModel = Model(inputs = input_layer_list, outputs = output_layer_list)
            yield parallelModel, n, batch

    def get_all_models(self):
        parallel_model_generator = self._parallel_model_generator()
        for parallelModel in parallel_model_generator:
            print(parallelModel.summary())

    def get_model_confs(self):
        model_confs = []
        # [16,32,64,128,256,512,1024]
        # [16,32,64,128,512,1024]
        s = search.Search_Space_Gen_1(node_options = [16,64,128], min_no_layers = 2, max_no_layers = self._max_no_layers, input_shape = self._input_shape)
        model_conf_batch = []

        # print(s.no_of_perm)
        for i,layer_no_sel in zip(range(s.min_no_layers, s.max_no_layers + 1), s.all_layer_perm):

            layer_no_models = []
            n = 0
            # print(layer_no_sel)

            for layer_list in layer_no_sel:
                layer_conf = s.get_layer_conf(layer_list)
                conf = ['', self._input_shape, i, "relu", layer_conf, [self._output_shape, self._output_activation]]
                # print(conf)
                model_conf_batch.append(conf)

                n = n + 1
                
                if(n == self._model_per_batch or layer_list == layer_no_sel[-1]):
                    # print(model_conf_batch)
                    model_confs.append(model_conf_batch)
                    model_conf_batch = []
                    n = 0
        
        self._model_confs = model_confs

    @staticmethod
    def get_layer_conf(layer_sizes):
        d = {}
        for layer_size, i in zip(layer_sizes, range(1,len(layer_sizes)+1)):
            layer_name = "layer" + str(i)
            d.update({layer_name:layer_size})

    @staticmethod
    def _get_input_output_layer_list(model_list):
        input_layer_list = []
        output_layer_list = []
        for nn_model_conf in model_list:
            nn_model = model_gen.NN_ModelGeneration(*nn_model_conf)
            input_layer_list.append(nn_model.input_layer)
            output_layer_list.append(nn_model.output_layer)
        return input_layer_list, output_layer_list

    def _get_train_lists(self, n):
        input_x = []
        input_labels = []
        input_test_x = []
        input_test_labels = []
        for i in range(n):
            input_x.append(self._train_x)
            input_labels.append(self._train_y)
            input_test_x.append(self._test_x)
            input_test_labels.append(self._test_y)
        return [input_x, input_labels, input_test_x, input_test_labels]