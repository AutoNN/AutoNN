from AutoNN.networkbuilding import multiple_model_gen_v3 as multiple
from AutoNN.networkbuilding import dataframe_extractor as de
from AutoNN.networkbuilding import model_generation as model_gen
from AutoNN.networkbuilding import model_optimization as model_opt
from AutoNN.networkbuilding import model_stacking

class Final:
    def __init__(self, train_x, train_y, test_x, test_y, loss_fn, epochs, batch_size, input_shape, output_shape = 1, output_activation = None, max_no_layers = 3, model_per_batch = 10, save_dir = ""):
        self._train_x = train_x
        self._train_y = train_y
        self._test_x = test_x
        self._test_y = test_y
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._loss_fn = loss_fn
        self._epochs = epochs
        self._batch_size = batch_size
        self._output_activation = output_activation
        self._max_no_layers = max_no_layers
        self._model_per_batch = model_per_batch
        self._save_dir = save_dir
        self._model_confs = []
        self._evaluate_dict_list = []
        self._no_top_model = 10

    def get_all_best_models(self):
        m = multiple.Multiple_Model_Gen_V3(self._train_x, self._train_y, self._test_x, self._test_y, self._loss_fn, self._epochs, self._batch_size, input_shape = self._input_shape, 
                                   max_no_layers = self._max_no_layers, model_per_batch = self._model_per_batch, output_shape = self._output_shape, output_activation = self._output_activation,
                                   save_dir = "/home/anish/AutoNN_test_weights/")
        m.get_best_models(save = False)
        opt = model_opt.Model_Optimization(self._train_x, self._train_y, self._test_x, self._test_y, self._loss_fn, 200, m.evaluate_dict_list, save_dir = "/home/anish/AutoNN_test_weights/candidate_models/")
        opt.optimize_models(save=True)
        model_history_list = []
        for model_dict in opt.evaluate_dict_list:
            model_history_list.append(model_dict["model_history"])
        stacked_models = model_stacking.Model_Stacking(self._train_x, self._train_y, self._test_x, self._test_y, self._loss_fn, opt.saved_paths, opt.opt_model_confs, save_dir = "/home/anish/AutoNN_test_weights/stacked_models/")
        stacked_models.optimize_stacked_models()
        return model_history_list
