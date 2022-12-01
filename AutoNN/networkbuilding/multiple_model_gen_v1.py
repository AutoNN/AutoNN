from AutoNN.networkbuilding import model_generation as model_gen
from AutoNN.networkbuilding.search_space_gen_v1 import Search_Space_Gen_1 as search
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

class Multiple_Model_Gen(search):

    def __init__(self, train_x, train_y, epochs, batch_size, min_no_layers = 2, max_no_layers = 5, input_shape = 8, output_shape = 1, model_per_batch = 10):
    # def __init__(self, min_no_layers = 2, max_no_layers = 5, input_shape = 8, output_shape = 1, model_per_batch = 10):
        # super().__init__(node_options = [16,32,64,128,196,256], min_no_layers = min_no_layers, max_no_layers = max_no_layers, input_shape = input_shape)
        # print(self._all_layer_perm)
        super().__init__(node_options = [16,32], min_no_layers = min_no_layers, max_no_layers = max_no_layers, input_shape = input_shape)
        self._train_x = train_x
        self._train_y = train_y
        self._epochs = epochs
        self._batch_size = batch_size
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._model_per_batch = model_per_batch

        
    def get_all_models(self):
        # Create Parallel Models (generator)
        parallel_model_generator = self._generate_parallel_model()
        for parallelModel,n in parallel_model_generator:
            print(parallelModel.summary())       
            # Train Model
            input_x = []
            input_labels = []
            adam_optimizer = Adam(lr = 1e-1)
            parallelModel.compile(loss = "mean_squared_error", optimizer = tf.keras.optimizers.Adam(), metrics = ["mean_absolute_error"])
            
            if(len(input_x) != n):
                input_x, input_labels = self._get_train_lists(n)
            # history = parallelModel.fit([self._train_x, self._train_x, self._train_x, self._train_x, self._train_x, self._train_x, self._train_x, self._train_x, self._train_x, self._train_x],
            #                         [self._train_y, self._train_y, self._train_y, self._train_y, self._train_y, self._train_y, self._train_y, self._train_y, self._train_y, self._train_y],
            #                         epochs=self._epochs, batch_size=self._batch_size
            #                         )
            history = parallelModel.fit(input_x, input_labels, epochs = self._epochs, batch_size = self._batch_size)
            # Evaluate Best Running Model
            scores = parallelModel.evaluate(input_x, input_labels, verbose = 0)

            print("\n")
            print("\n")
            print("\n")

            for name,score in zip(parallelModel.metrics_names, scores):
                print(name, " : ", score)

            print("\n------------------------------------------------------------------------------------------------------------\n")
            # Store Weights of Best model
            # Clear Keras Backend and set n = 0

    def _generate_parallel_model(self):

        # tf.keras.backend.clear_session()

        # Selecting Layer Number ie 2 Hidden layers or 3 Hidden Layers or ....
        for i,layer_no_sel in zip(range(self._min_no_layers, self._max_no_layers + 1), self._all_layer_perm):

            layer_no_models = []
            n = 0

            for layer_list in layer_no_sel:
                layer_conf = self.get_layer_conf(layer_list)
                layer_no_models.append(model_gen.NN_ModelGeneration(input_shape = self._input_shape, init_no_layers = i, init_activation_fn = "relu", init_layer_conf = layer_conf, output_layer_conf = [1,"softmax"]))
                
                n = n + 1
                
                if(n == self._model_per_batch or layer_list == layer_no_sel[-1]):

                    input_layer_list, output_layer_list = self._get_input_output_layer_list(layer_no_models)
                    parallelModel = Model(inputs = input_layer_list, outputs = output_layer_list)
                    # parallelModel = layer_no_models[0].model
                    # print(parallelModel.summary())
                    # print(len(parallelModel.input))
                    yield parallelModel, len(layer_no_models)

                    layer_no_models = []
                    # tf.keras.backend.clear_session()
                    n = 0
                    # break
            # break            

            # self._all_model_list.append(layer_no_models)

    
    @staticmethod
    def _get_input_output_layer_list(model_list):
        input_layer_list = []
        output_layer_list = []
        for nn_model in model_list:
            input_layer_list.append(nn_model.input_layer)
            output_layer_list.append(nn_model.output_layer)
        return input_layer_list, output_layer_list

    def _get_train_lists(self, n):
        input_x = []
        input_labels = []
        for i in range(n):
            input_x.append(self._train_x)
            input_labels.append(self._train_y)
        return input_x, input_labels
