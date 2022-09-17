import itertools
# from ASC_ML.networkbuilding import model_generation as model_gen

class Search_Space_Gen_1:

    def __init__(self, node_options = [16,32,64,128,196,256], min_no_layers = 2, max_no_layers = 5, input_shape = 8):
        self._node_options = node_options
        self._all_layer_perm = []
        self._min_no_layers = min_no_layers
        self._max_no_layers = max_no_layers
        self._get_all_layer_permutations()
        self._input_shape = input_shape
        self._no_of_perm = 0

    # def _get_all_layer_permutations(self):
    #     for i in range(self._min_no_layers, self._max_no_layers+1):
    #         layer_choice = []
    #         for j in range(i):
    #             layer_choice.append(self._node_options)
    #         self._all_layer_perm.append(list(itertools.product(*layer_choice)))

    def _get_all_layer_permutations(self):
        for i in range(self._min_no_layers, self._max_no_layers+1):
            layer_choice = []
            for j in range(i):
                layer_choice.append(self._node_options)
            append_layer_arr = list(itertools.product(*layer_choice))
            append_layer_arr_clean = []
            flag = True
            
            # filter out redundant model configurations
            if i > 2:
                for model_arch in append_layer_arr:
                    for layer,pos in zip(model_arch[1:-1], range(1,len(model_arch)-1)):
                        if model_arch[pos-1] > layer and model_arch[pos+1] > layer:
                            flag = False
                            # print("Dropped ", model_arch)
                    if flag:
                        append_layer_arr_clean.append(model_arch)
                    flag = True                
            else: 
                append_layer_arr_clean = append_layer_arr            
            self._all_layer_perm.append(append_layer_arr_clean)

    @property
    def all_layer_perm(self):
        # Return all layer permutations
        return self._all_layer_perm

    @property
    def no_of_perm(self):
        # Return number of permutations
        for model_list in self._all_layer_perm:
            self._no_of_perm += len(model_list)
        return self._no_of_perm
    
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