from dask import dataframe as dd
import numpy as np
from dask import array as da
from dask_ml.preprocessing import DummyEncoder
from dask_ml.preprocessing import Categorizer
from sklearn.pipeline import make_pipeline

class Encoding:
    def __init__(self):
        self._key_dict_enc = {}
        self._key_dict_dec = {}
        self._one_hot_pipe = None
        self._cardinality_threshold = 4

    @property
    def encode_keys(self):
        return self._key_dict_enc

    @property
    def decode_keys(self):
        return self._key_dict_dec


    def fit_column(self, column_name, column, label_name):
        no_keys = column.unique()
        self._keyval(column_name, no_keys, label_name)


    def _keyval(self, col_name, no_keys, label_name):
        no_keys = list(enumerate(no_keys, start=1))
        inverse_keyvalpairs = dict([(0,'UnknownData')]+no_keys)
        keyvalpairs = list(map(lambda x: (x[1],x[0]), no_keys))
        keyvalpairs = dict(keyvalpairs)
        keyvalpairs[np.nan] = np.nan
        self._key_dict_enc.update({col_name:keyvalpairs})
        if len(no_keys) <= self._cardinality_threshold:
            if col_name == label_name and len(no_keys) != 2:
                self._key_dict_dec.update({col_name:inverse_keyvalpairs})
            elif col_name != label_name:
                self._key_dict_dec.update({col_name:inverse_keyvalpairs})

    def onehot_fit(self, dataframe):
        self._one_hot_pipe = make_pipeline(
            Categorizer(),
            DummyEncoder()
        )
        self._one_hot_pipe.fit(dataframe)
    
    def onehot_encode(self, dataframe):
        dataframe_onehot = self._one_hot_pipe.transform(dataframe)
        return dataframe_onehot
    
    def label_encode(self, dataframe):
        dataframe_encoded = dataframe.replace(self._key_dict_enc).copy()
        return dataframe_encoded

    def inverse_label_encode(self, dataframe):
        dataframe_decoded = dataframe.replace(self._key_dict_dec).copy()
        return dataframe_decoded