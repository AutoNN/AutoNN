from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Concatenate, Flatten, Input
from tensorflow.keras.activations import tanh, relu, sigmoid, softmax, exponential


class NN_ModelGeneration:

    def __init__(self, model_name = "model_1", input_shape = 8, init_no_layers = 1, init_activation_fn = "relu", init_layer_conf = {"layer1":8}, output_layer = [1,"softmax"]):
        self._model_name = model_name
        self._input_shape = input_shape
        self._no_of_layers = init_no_layers
        self._activation_fn = init_activation_fn
        self._layer_conf = init_layer_conf
        self._output_layer = output_layer
        self._model, self._input_layer, self._penultimate_layer = self.generate_model()

    @property
    def model_name(self):
        # Return Model Name
        return self._model_name
    
    @property
    def model_specs(self):
        # Return Model Specifications
        return [self._input_shape, self._no_of_layers, self._activation_fn, self._layer_conf]
    
    @property
    def model(self):  
        # Return Model    
        return self._model

    def generate_model(self):
        # Generate Initial Model from above given initial parameters

        #Check _no_of_layers matches num of items in _layer_conf
        self._check_conf(self._no_of_layers, self._layer_conf)

        input_layer = Input(self._input_shape, name = 'input_layer')
        x = input_layer
        for layer_name in self._layer_conf:
            x = Dense(self._layer_conf[layer_name], activation = self._activation_fn, name = layer_name)(x)

        output_layer = Dense(self._output_layer[0], activation = self._output_layer[1], name = "output_layer")(x)

        model = Model(inputs = input_layer, outputs = [output_layer], name = self._model_name)
        return model,input_layer, x

    def append_model(self, no_layers = 1, layer_conf = {"layer_append1":8}):
        # Append Layers to initially created model
        
        self._layer_conf.update(layer_conf)
        self._no_of_layers = self._no_of_layers + no_layers
        self._check_conf(self._no_of_layers, self._layer_conf)

        x = self._penultimate_layer
        for layer_name in layer_conf:
            x = Dense(layer_conf[layer_name], activation = self._activation_fn, name = layer_name)(x)
        
        output_layer = Dense(self._output_layer[0], activation = self._output_layer[1], name = "output_layer")(x)
        self._model = Model(inputs = self._input_layer, outputs = [output_layer], name = self._model_name)
        self._penultimate_layer = x


    @staticmethod
    def _check_conf(no_of_layers, layer_conf):
        if(no_of_layers != len(layer_conf)):
            raise AssertionError(f"Layer Configuration Dict length {len(layer_conf)} is not equal to No. of Layers {no_of_layers}!")

