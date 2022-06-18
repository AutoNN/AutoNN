import itertools
from ASC_ML import model_generation as model_gen

class Search_Space_Gen_1:

    def __init__(self, node_options = [16,32,64,128,196,256], min_no_layers = 2, max_no_layers = 5, input_shape = 8):
        self._node_options = node_options
        self._all_layer_perm = []
        self._min_no_layers = min_no_layers
        self._max_no_layers = max_no_layers
        self._get_all_layer_permutations()
        self._input_shape = input_shape

    def _get_all_layer_permutations(self):
        for i in range(self._min_no_layers, self._max_no_layers+1):
            layer_choice = []
            for j in range(i):
                layer_choice.append(self._node_options)
            self._all_layer_perm.append(list(itertools.product(*layer_choice)))

    @property
    def all_layer_perm(self):
        # Return Model Name
        return self._all_layer_perm
    
    @property
    def min_no_layers(self):
        # Return Model Name
        return self._min_no_layers
    
    @property
    def max_no_layers(self):
        # Return Model Name
        return self._max_no_layers

    @staticmethod
    def get_layer_conf(layer_list):
        layer_conf = {}
        for i,layer_no in zip(range(1,len(layer_list)+1), layer_list):
            layer_name = "layer"+str(i)
            curr_layer = {layer_name : layer_no}
            layer_conf.update(curr_layer)
        return layer_conf