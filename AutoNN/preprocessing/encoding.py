from dask import dataframe as dd

class Encoding:
    def __init__(self):
        self._key_dict = {}

    @property
    def key_dict(self):
        return self._key_dict

    def send_column(self, column = [], col_name = ""):
        no_keys = column.unique()
        keyvalpairs = self._keyval(col_name, no_keys)

    def encode(self, dataframe:dd, enc_type = "label"):
        if enc_type == "label":
            dataframe_encoded = dataframe.replace(self._key_dict).copy()
        return dataframe_encoded

    def _keyval(self, col_name, no_keys):
        i = 0
        keyvalpairs = list()
        for key in no_keys:
            keyvalpairs.append((key,i))
            i+=1
        keyvalpairs = dict(keyvalpairs)
        self._key_dict.update({col_name:keyvalpairs})

    def onehot_enc():
        pass
    
    def label_enc():

        pass
