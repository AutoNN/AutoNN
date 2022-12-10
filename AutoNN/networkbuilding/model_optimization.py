from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import RandomUniform, GlorotUniform, GlorotNormal, HeUniform, HeNormal, LecunUniform, LecunNormal
from tensorflow.keras.activations import tanh, relu, selu
from AutoNN.networkbuilding import hyperparameter_optimization as hyp_opt
from AutoNN.networkbuilding import model_generation as model_gen

import os

class Model_Optimization:

    def __init__(self, train_x, train_y, test_x, test_y, loss_fn, epochs, model_dict_list, save_dir):
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
        self._epochs = epochs
        self._model_dict_list = model_dict_list
        self._save_dir = save_dir
        self._saved_paths = []
        self._model_confs = []
        self._opt_model_confs = []
        self._best_hyp_permodel = []
        self._no_top_model = 2
        self._evaluate_dict_list = []

    @property
    def saved_paths(self):
        return self._saved_paths
    
    @property
    def model_confs(self):
        return self._model_confs
    
    @property
    def opt_model_confs(self):
        return self._opt_model_confs

    @property
    def best_hyp_permodel(self):
        return self._best_hyp_permodel
    
    @property
    def evaluate_dict_list(self):
        return self._evaluate_dict_list
    
    def _candidate_model_generator(self):
        for model_dict in self._model_dict_list:
            model_conf = model_dict["model_conf"]
            model_conf.append(True)
            input_layer_list, output_layer_list = self._get_input_output_layer_list([model_conf])
            model = Model(name = model_dict["model_name"]+"_dr",inputs = input_layer_list, outputs = output_layer_list)
            # model = load_model(model_dict["path_weights"])
            self._model_confs.append(model_conf)
            yield model, model_conf
    
    def optimize_models(self, save):
        candidate_model_generator = self._candidate_model_generator()
        for model, model_conf in candidate_model_generator:
            # print(model.summary())
            print("--------------------------------------------------------------------------------")
            print(f"Model Name : {model.name}\n")
            h = hyp_opt.Hyperparameter_Optimization([self._train_x], [self._train_y], [self._test_x], [self._test_y], model, self._loss_fn, dropout_opt = True)
            best_lr, best_batch_size, best_activation, best_initializer, best_dropout_rate = h.get_best_hyperparameters()

            self._reinitialize_model(model, best_initializer)
            self._set_activation(model, best_activation)
            if best_dropout_rate != None:
                model.layers[-2].rate = best_dropout_rate
            self._best_hyp_permodel.append([best_lr, best_batch_size, best_activation, best_initializer, best_dropout_rate])

            model, metrics_names, scores, scores_test, history = self.train_model(input_data = [self._train_x, self._train_y, self._test_x, self._test_y], Pmodel=model, n_model=1,
                                epochs=100,lr=best_lr, batch_size=best_batch_size)
            self._evaluate_save_model(Model = model, model_conf = model_conf, metrics_names=metrics_names, scores=scores_test, model_history = history)
            # if save==True:
            #     self.save_weights(model)
        print(self._evaluate_dict_list)
        # self.save_weights()
    
    # def save_weights(self, model):
    #     # save_dir = os.path.join(self._save_dir,"/candidate_models")
    #     save_path = os.path.join(self._save_dir, model.name)
    #     # if os.path.isdir(save_dir):
    #     #     os.mkdir(save_dir)
    #     model.save(save_path)
    #     self._saved_paths.append(save_path)

    def save_weights(self):
        for dicti in self._evaluate_dict_list:
            model = dicti["model"]
            model.save(dicti["path_weights"])
            self._saved_paths.append(dicti["path_weights"])
            self._opt_model_confs.append(dicti["model_conf"])
                
    def train_model(self, input_data, Pmodel, n_model, epochs, lr = 1e-3, batch_size = 64, activation = None, initializer = None):

        # if activation != None:
        #     Pmodel = self._set_activation(Pmodel, activation)
        # if initializer != None:    
        #     Pmodel = self._reinitialize_model(Pmodel, initializer)

        input_x, input_labels, input_test_x, input_test_labels = input_data
        optimizer = Adam(learning_rate = lr)
        Pmodel.compile(loss = self._loss_fn, optimizer = optimizer)
        history = Pmodel.fit(input_x, input_labels, validation_data = (input_test_x, input_test_labels), epochs = epochs, batch_size = batch_size, verbose = 0)

        metrics_names, scores, scores_test = self.print_scores(Pmodel, input_data)
        
        return Pmodel, metrics_names, scores, scores_test, history

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

    def _evaluate_save_model(self, Model, model_conf, metrics_names, scores, model_history):
        # n : Number of models in the Model
        # List of dictionaries {"model_name":"densexyz", "score":0.000, "path_weights":"/home/something/dense"}

        metrics_name = metrics_names[0]
        model_score = scores[0]
        entry_flag = False

        model_name = Model.name
        model_name = model_name.removesuffix("_loss")
        curr_model_dict = {"model_name":model_name, "score":model_score, "path_weights":os.path.join(self._save_dir, model_name), 
                            "model_conf":model_conf, "model":Model, "model_history": model_history}

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
        for layer in model.layers[1:-1]:
            layer.activation = activation