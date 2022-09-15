from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Concatenate, Flatten, Input, Dropout
from tensorflow.keras.activations import tanh, relu, sigmoid, softmax, exponential


class NN_ModelGeneration:

    def __init__(self, model_name = "model_1", input_shape = 8, init_no_layers = 1, init_activation_fn = "relu", init_layer_conf = {"layer1":8}, output_layer_conf = [1,None], dr = False):
        # self._model_name = model_name
        self._model_name = self._get_model_name(init_layer_conf, dr)
        self._input_shape = input_shape
        self._no_of_layers = init_no_layers
        self._activation_fn = init_activation_fn
        self._layer_conf = init_layer_conf
        self._layer_conf = {k + "_" + self._model_name: v for k, v in self._layer_conf.items()}
        self._output_layer_conf = output_layer_conf
        self._dr = dr
        
        self._model, self._input_layer, self._penultimate_layer, self._output_layer = self.generate_model()

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
    
    @property
    def input_layer(self):
        # Return Input Layer
        return self._input_layer

    @property
    def output_layer(self):
        # Return Input Layer
        return self._output_layer
    
    @property
    def output_layer_conf(self):
        # Return Input Layer
        return self._output_layer_conf

    @property
    def layer_conf(self):
        # Return Layer Conf
        return self._layer_conf

    def generate_model(self):
        # Generate Initial Model from above given initial parameters

        #Check _no_of_layers matches num of items in _layer_conf
        self._check_conf(self._no_of_layers, self._layer_conf)

        input_layer = Input(self._input_shape, name = 'input_layer' + "_" + self._model_name)
        x = input_layer
        for layer_name in self._layer_conf:
            x = Dense(self._layer_conf[layer_name], activation = self._activation_fn, name = layer_name)(x)

        if self._dr == True:
            x = Dropout(0.0)(x)

        output_layer = Dense(self._output_layer_conf[0], activation = self._output_layer_conf[1], name = "output_layer" + "_" + self._model_name)(x)

        model = Model(inputs = input_layer, outputs = [output_layer], name = self._model_name)
        return model, input_layer, x, output_layer

    def append_model(self, no_layers = 1, layer_conf = {"layer_append1":8}):
        # Append Layers to initially created model
        
        self._layer_conf.update(layer_conf)
        self._no_of_layers = self._no_of_layers + no_layers
        self._check_conf(self._no_of_layers, self._layer_conf)

        x = self._penultimate_layer
        for layer_name in layer_conf:
            x = Dense(layer_conf[layer_name], activation = self._activation_fn, name = layer_name)(x)
        
        output_layer = Dense(self._output_layer_conf[0], activation = self._output_layer_conf[1], name = "output_layer")(x)
        self._model = Model(inputs = self._input_layer, outputs = [output_layer], name = self._model_name)
        self._penultimate_layer = x

    @staticmethod
    def _check_conf(no_of_layers, layer_conf):
        if(no_of_layers != len(layer_conf)):
            raise AssertionError(f"Layer Configuration Dict length {len(layer_conf)} is not equal to No. of Layers {no_of_layers}!")

    @staticmethod
    def _get_model_name(layer_conf, dr):
        a = "dense_"
        for value in layer_conf.values():
            a = a + str(value) + "_"
        a = a[:-1]
        if dr == True:
            a = a+"_dr"
        return a
