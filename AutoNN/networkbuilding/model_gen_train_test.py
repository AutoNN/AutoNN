from AutoNN.networkbuilding import model_generation as model_gen
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np

class Model_Parallel_Train_Test:

    def __init__(self, train_x, train_y, input_shape, output_shape, output_activation):
        self._train_x = train_x
        self._train_y = train_y
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._output_activation = output_activation
        self._model_archs = [['model_4n', self._input_shape, 3, "relu", {"layer1":4, "layer2":4, "layer3":4}, [output_shape, output_activation]],
                             ['model_8n', self._input_shape, 3, "relu", {"layer1":8, "layer2":8, "layer3":8}, [output_shape, output_activation]], 
                             ['model_16n', self._input_shape, 3, "relu", {"layer1":16, "layer2":16, "layer3":16}, [output_shape, output_activation]],
                             ['model_32n', self._input_shape, 3, "relu", {"layer1":32, "layer2":32, "layer3":32}, [output_shape, output_activation]],
                             ['model_64n', self._input_shape, 3, "relu", {"layer1":64, "layer2":64, "layer3":64}, [output_shape, output_activation]],
                             ['model_128n', self._input_shape, 3, "relu", {"layer1":128, "layer2":128, "layer3":128}, [output_shape, output_activation]]
                             ]

    def get_models(self):
        # nn_model_4n = model_gen.NN_ModelGeneration(*self._model_archs[0])
        # nn_model_8n = model_gen.NN_ModelGeneration(*self._model_archs[1])
        # nn_model_16n = model_gen.NN_ModelGeneration(*self._model_archs[2])
        # nn_model_32n = model_gen.NN_ModelGeneration(*self._model_archs[3])
        # nn_model_64n = model_gen.NN_ModelGeneration(*self._model_archs[4])
        # nn_model_128n = model_gen.NN_ModelGeneration(*self._model_archs[5])
        nn_model_list = []
        for i in range(6):
            nn_model_list.append(model_gen.NN_ModelGeneration(*self._model_archs[i]))

        # parallelModel = Model(inputs = [nn_model_4n.input_layer, nn_model_8n.input_layer, nn_model_16n.input_layer, nn_model_32n.input_layer, nn_model_64n.input_layer, nn_model_128n.input_layer], 
        #                         outputs = [nn_model_4n.output_layer, nn_model_8n.output_layer, nn_model_16n.output_layer, nn_model_32n.output_layer, nn_model_64n.output_layer, nn_model_128n.output_layer])
        # parallelModel = Model(inputs = [nn_model_list[0].input_layer, nn_model_list[1].input_layer, nn_model_list[2].input_layer, nn_model_list[3].input_layer, nn_model_list[4].input_layer, nn_model_list[5].input_layer], 
        #                         outputs = [nn_model_list[0].output_layer, nn_model_list[1].output_layer, nn_model_list[2].output_layer, nn_model_list[3].output_layer, nn_model_list[4].output_layer, nn_model_list[5].output_layer])
        input_layer_list, output_layer_list = self._get_input_output_layer_list(nn_model_list)
        parallelModel = Model(inputs = input_layer_list, outputs = output_layer_list)  

        return parallelModel, nn_model_list

    def train_models(self):
        parallelModel, nn_model_list = self.get_models()

        adam_optimizer = Adam(learning_rate = 1e-3)

        parallelModel.compile(loss = "mean_squared_error",
             optimizer = adam_optimizer,
             metrics = ["mean_absolute_error"])

        history = parallelModel.fit([self._train_x, self._train_x, self._train_x, self._train_x, self._train_x, self._train_x],
                                    [self._train_y, self._train_y, self._train_y, self._train_y, self._train_y, self._train_y],
                                    epochs=100, batch_size=128
                                    )
        # history = parallelModel.fit(self._train_x, self._train_y, epochs=80, batch_size=64
        #                             )

        scores = parallelModel.evaluate([self._train_x, self._train_x, self._train_x, self._train_x, self._train_x, self._train_x],
                                    [self._train_y, self._train_y, self._train_y, self._train_y, self._train_y, self._train_y],
                                    verbose = 0
                                    )
                            
        print("\n")
        print("\n")
        print("\n")

        for name,score in zip(parallelModel.metrics_names, scores):
            print(name, " : ", score)

        # indi_scores = nn_model_list[0].model.evaluate([self._train_x], [self._train_y])
        # print("1st model score : ", indi_scores)

        return parallelModel

    @staticmethod
    def _get_input_output_layer_list(model_list):
        input_layer_list = []
        output_layer_list = []
        for nn_model in model_list:
            input_layer_list.append(nn_model.input_layer)
            output_layer_list.append(nn_model.output_layer)
        return input_layer_list, output_layer_list
        
