from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout

from ASC_ML.networkbuilding import model_generation as model_gen
from ASC_ML.networkbuilding import hyperparameter_optimization as hyp_opt

class Model_Stacking:
    def __init__(self, train_x, train_y, test_x, test_y, model_path_list, model_conf_list):
        self._model_path_list = model_path_list
        self._model_conf_list = model_conf_list
        self._train_x = train_x
        self._train_y = train_y
        self._test_x = test_x
        self._test_y = test_y
    
    def get_loss_function(self):
        # Logic to get loss funtion
        # return "mean_absolute_percentage_error"
        # return self.root_mean_squared_error
        # return "mean_squared_error"
        return "mean_absolute_error"

    def _stacked_model_generator(self):
        for path in self._model_path_list:
            for model_conf in self._model_conf_list:
                model1 = load_model(path)
                activation = model1.layers[-2].get_config()["activation"]
                reduced_model1 = Model(name = model1.name+"_reduced", inputs = model1.input, outputs = model1.layers[-2].output)
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
        loss_fn = self.get_loss_function()
        stacked_model_generator = self._stacked_model_generator()
        for model in stacked_model_generator:
            # print(model.summary())
            print(model.name)
            h = hyp_opt.Hyperparameter_Optimization([self._train_x], [self._train_y], model, loss_fn, activation_opt = False, initializer_opt = False)
            best_lr, best_batch_size, _, _ = h.get_best_hyperparameters()
            print(f"BEST LR STACKED {best_lr}, BEST BATCH SIZE {best_batch_size}")

