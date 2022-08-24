from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout

from ASC_ML.networkbuilding import model_generation as model_gen

class Model_Stacking:
    def __init__(self, model_path_list, model_conf_list):
        self._model_path_list = model_path_list
        self._model_conf_list = model_conf_list

    def _stacked_model_generator(self):
        for path in self._model_path_list:
            model1 = load_model(path)
            activation = model1.layers[-2].get_config()["activation"]
            reduced_model1 = Model(name = model1.name+"_reduced", inputs = model1.input, outputs = model1.layers[-2].output)
            last_layer = reduced_model1.output
            x = last_layer
            for model_conf in self._model_conf_list:
                model2_obj = model_gen.NN_ModelGeneration(*model_conf)
                x = Dropout(0.0)
                for layer_name in model2_obj.layer_conf:
                    x = Dense(self._layer_conf[layer_name], activation = activation, name = layer_name)(x)
                x = Dropout(0.0)(x)
                output_layer = Dense(model2_obj.output_layer_conf[0], activation = model2_obj.output_layer_conf[1], name = "output_layer" + "_" + model2_obj.model_name)(x)

                stacked_model = Model(inputs = reduced_model1.input, outputs = x)
                yield stacked_model

    def stacked_models(self):
        stacked_model_generator = self._stacked_model_generator()
        for model in stacked_model_generator:
            print(model.summary())
