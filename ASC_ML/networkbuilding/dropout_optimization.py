from tensorflow.keras.optimizers import Adam

class Dropout_Optimization():
    def __init__(self, train_x, train_y, test_x, test_y, epochs, model):
        self._train_x = train_x
        self._train_y = train_y
        self._test_x = test_x
        self._test_y = test_y
        self._epochs = epochs
        self._model = model

    def dropout_optimization(self, test_X, test_Y, lr = 1e-3, batch_size = 64, epoch = 100):
        dropout_indices = self._get_dropout_index()
        dropout_list = [0,0.2,0.5]
        dropout_losses = []
        for dropout0 in dropout_list:
            for dropout1 in dropout_list:
                self._model.layers[dropout_indices[0]].rate = dropout0
                self._model.layers[dropout_indices[1]].rate = dropout1
                optimizer = Adam(lr = lr)
                self._model.compile(loss = self._loss_fn, optimizer = optimizer)
                history = self._model.fit(self._train_X, self._train_Y, epochs = epoch, batch_size = batch_size, verbose = 0)

                scores = self._model.evaluate(self._train_X, self._train_Y, verbose = 0)
                scores_test = self._model.evaluate(test_X, test_Y, verbose = 0)
                print(f"For Dropouts D0 = {dropout0}, D1 = {dropout1}, TRAIN_LOSS = {scores}, TEST_LOSS = {scores_test}")

    def _get_dropout_index(self):
        dropout_indices = []
        ind = 0
        for layer in self._model.layers:
            if layer.__class__.__name__ == "Dropout":
                dropout_indices.append(ind)
            ind = ind+1
        return dropout_indices