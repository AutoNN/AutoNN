from dask import dataframe as dd
import numpy as np
from dask import array as da

class Encoding:
    def __init__(self):
        self._key_dict_enc = {}
        self._key_dict_dec = {}

    @property
    def encode_keys(self):
        return self._key_dict_enc

    @property
    def decode_keys(self):
        return self._key_dict_dec


    def fit_column(self, column_name, column):
        no_keys = column.unique()
        self._keyval(column_name, no_keys)


    def _keyval(self, col_name, no_keys):
        no_keys = list(enumerate(no_keys, start=1))
        inverse_keyvalpairs = dict([(0,'UnknownData')]+no_keys)
        keyvalpairs = list(map(lambda x: (x[1],x[0]), no_keys))
        keyvalpairs = dict(keyvalpairs)
        keyvalpairs[np.nan] = np.nan
        self._key_dict_enc.update({col_name:keyvalpairs})
        self._key_dict_dec.update({col_name:inverse_keyvalpairs})

    def onehot_enc():
        pass
    
    def label_encode(self, dataframe):
        dataframe_encoded = dataframe.replace(self._key_dict_enc).copy()
        return dataframe_encoded

    def inverse_label_encode(self, dataframe):
        dataframe_decoded = dataframe.replace(self._key_dict_dec).copy()
        return dataframe_decoded
