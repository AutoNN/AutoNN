from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform, GlorotUniform, GlorotNormal, HeUniform, HeNormal, LecunNormal, LecunUniform
from tensorflow.keras.activations import tanh, relu, selu

class Dropout_Optimization():
    def __init__(self, train_x, train_y, test_x, test_y, loss_fn, epochs, model):
        self._train_x = train_x
        self._train_y = train_y
        self._test_x = test_x
        self._test_y = test_y
        self._loss_fn = loss_fn
        self._epochs = epochs
        self._model = model
    

    def dropout_optimization(self, lr = 1e-3, batch_size = 64, activation = "relu", initializer = "RandomUniform", epoch = 75):
        dropout_indices = self._get_dropout_index()
        dropout_list = [0,0.2,0.5]
        dropout_comb_list = []
        # dropout_loss_list = []
        dropout_loss_diff_list = []
        for dropout0 in dropout_list:
            for dropout1 in dropout_list:
                self._model.layers[dropout_indices[0]].rate = dropout0
                self._model.layers[dropout_indices[1]].rate = dropout1
                dropout_comb_list.append([dropout0,dropout1])

                self._reinitialize_model(self._model, initializer)
                self._set_activation(self._model, activation)
                optimizer = Adam(learning_rate = lr)
                self._model.compile(loss = self._loss_fn, optimizer = optimizer)
                history = self._model.fit(self._train_x, self._train_y, epochs = epoch, batch_size = batch_size, verbose = 0)

                scores = self._model.evaluate(self._train_x, self._train_y, verbose = 0)
                scores_test = self._model.evaluate(self._test_x, self._test_y, verbose = 0)
#                 print(f"For Dropouts D0 = {dropout0}, D1 = {dropout1}, TRAIN_LOSS = {scores}, TEST_LOSS = {scores_test}")
                dropout_loss_diff_list.append(scores_test - scores)
                
        # best_dropout_rates = dropout_comb_list[dropout_loss_list.index(min(dropout_loss_list))]
        best_dropout_rates = dropout_comb_list[dropout_loss_diff_list.index(min(dropout_loss_diff_list))]
        self._model.layers[dropout_indices[0]].rate = best_dropout_rates[0]
        self._model.layers[dropout_indices[1]].rate = best_dropout_rates[1]
        return best_dropout_rates
                

    def _get_dropout_index(self):
        dropout_indices = []
        ind = 0
        for layer in self._model.layers:
            if layer.__class__.__name__ == "Dropout":
                dropout_indices.append(ind)
            ind = ind+1
        return dropout_indices

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