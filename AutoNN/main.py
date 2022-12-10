import numpy as np
import dask.dataframe as dd
from AutoNN.preprocessing import data_cleaning
from AutoNN.preprocessing import encoding_v3 as enc
from AutoNN.networkbuilding import final

class Autonn:
    def __init__(self, train_csv_path, label_name, epochs = 100, loss = None, save_path = None):
        self._train_csv_path = train_csv_path
        self._label_name = label_name
        self._output_shape = None
        self._output_activation = None
        self._loss = loss
        self._save_path = save_path
        self._epochs = epoch
        
        self._train_X = None
        self._train_Y = None
        self._test_X = None
        self._test_Y = None
        
        self._input_shape = None
        
        self._EDA_data_container = None

        self._history_list = []

        self._f = None
    
    @property
    def history_list(self):
        return self._history_list
    
    def preprocessing(self):
        df = dd.read_csv(self._train_csv_path, assume_missing=True, sample_rows=2000)
        d_clean = data_cleaning.DataCleaning(label = [self._label_name], train_dataframe=df)
        d_clean.dataset.train_test_split()
        d_clean.parse_dates()
        d_clean.generate_column_info()
        # d_clean.column_info
        encoder = enc.Encoding()

        train, validation, test = d_clean.dataset.get(['train', 'validation', 'test'])

        for column in d_clean.column_info.keys():
            if d_clean.column_info[column]['dtype'] == 'object':
                encoder.fit_column(column = train[column], column_name=column, label_name = self._label_name)
        train, validation, test = d_clean.dataset.get(['train', 'validation', 'test'])
        d_clean.dataset.set(encoder.label_encode(train), type = 'train')

        if validation is not None:
            d_clean.dataset.set(encoder.label_encode(validation), type = 'validation')
        if test is not None:
            d_clean.dataset.set(encoder.label_encode(test), type = 'test')


        d_clean.clean_data()
        
        self._EDA_data_container = d_clean.dataset
        
        d_clean.feature_elimination_fit(type = "train", method = "correlation")
        d_clean.eliminate_features(type = "train")
        if validation is not None:
            d_clean.eliminate_features(type = "validation")
        if test is not None:
            d_clean.eliminate_features(type = "test")
            

        train, validation, test = d_clean.dataset.get(['train', 'validation', 'test'])
# #         EDA DATA
#         self._EDA_train = train
#         self._EDA_test = test
        
        d_clean.dataset.set(encoder.inverse_label_encode(train), type = 'train')
        if validation is not None:
            d_clean.dataset.set(encoder.inverse_label_encode(validation), type = 'validation')
        if test is not None:
            d_clean.dataset.set(encoder.inverse_label_encode(test), type = 'test')
    
        train, validation, test = d_clean.dataset.get(['train', 'validation', 'test'])
        d_clean.generate_column_info()
        
        reg_clas_flag, label_cardinality = d_clean.is_regression()
        categorical_flag = False
        if reg_clas_flag == 1:
            self._output_shape = 1
            self._output_activation = None
            loss = "mean_squared_error"
        elif reg_clas_flag == 0 and label_cardinality == 2:
            self._output_shape = 1
            self._output_activation = "sigmoid"
            loss = "binary_crossentropy"
        elif reg_clas_flag == 0 and label_cardinality > 2:
            self._output_shape = label_cardinality
            self._output_activation = "softmax"
            loss = "categorical_crossentropy"
            categorical_flag = True
            print("CATEGORICAL")
#             d_clean.dataset.set(train.categorize(self._label_name), type = 'train')        
            
        if self._loss == None:
            self._loss = loss
            
        train, validation, test = d_clean.dataset.get(['train', 'validation', 'test'])
                    
        train_Y = train[self._label_name].to_frame()
        test_Y = test[self._label_name].to_frame()
        
        train_X = train.drop(self._label_name, axis = 1).copy()
        test_X = test.drop(self._label_name, axis = 1).copy()
        
        onehot_encoder_X = enc.Encoding()
        onehot_encoder_X.onehot_fit(train_X)
        
        train_X = onehot_encoder_X.onehot_encode(train_X)
        if validation is not None:
            d_clean.dataset.set(onehot_encoder_X.onehot_encode(validation), type = 'validation')
        if test is not None:
            test_X = onehot_encoder_X.onehot_encode(test_X)
        
        if categorical_flag == True:
            train_Y = train_Y.categorize(columns = [self._label_name])
            test_Y = test_Y.categorize(columns = [self._label_name])
            onehot_encoder_Y = enc.Encoding()
            onehot_encoder_Y.onehot_fit(train_Y)
            train_Y = onehot_encoder_Y.onehot_encode(train_Y)
            if test is not None:
                test_Y = onehot_encoder_Y.onehot_encode(test_Y)
                        
        d_clean.scaling_fit(train_X)
        train_X = d_clean.scaling_transform(train_X)
        if test is not None:
            test_X = d_clean.scaling_transform(test_X)
        
        if not categorical_flag:
            d_clean.scaling_fit(train_Y)
            train_Y = d_clean.scaling_transform(train_Y)
            if test is not None:
                test_Y = d_clean.scaling_transform(test_Y)
        
        self._train_X = np.asarray(train_X)
        self._train_Y = np.asarray(train_Y)
        self._test_X = np.asarray(test_X)
        self._test_Y = np.asarray(test_Y)
        print(self._train_Y.shape)
        
        self._input_shape = train_X.shape[-1]
        
#         return train, validation, test, train_Y, test_Y

    def neuralnetworkgeneration(self):
        '''
        train_x: Any,
        train_y: Any,
        test_x: Any,
        test_y: Any,
        loss_fn: Any,
        epochs: Any,
        batch_size: Any, 
        input_shape: Any,
        output_shape: int = 1,
        output_activation: Any | None = None,
        max_no_layers: int = 3,
        model_per_batch: int = 10,
        save_dir: str = ""

        '''
        self._f = final.Final(
        self._train_X,
         self._train_Y,
          self._test_X,
           self._test_Y,
            self._loss,
             self._epochs,
              64,
               input_shape = self._input_shape, 
                max_no_layers = 3, 
                model_per_batch = 10, 
                output_shape = self._output_shape, output_activation = self._output_activation,
                save_dir = self._save_path)
        self._history_list = self._f.get_all_best_models()
    
    def get_stacked_models(self):
        self._f.get_all_best_stacked()

    def save_candidate_models(self):
        self._f.save_model()
    
    def save_stacked_models(self):
        self._f.save_stacked_model()
