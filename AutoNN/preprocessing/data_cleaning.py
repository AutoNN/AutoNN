from dask import dataframe as dd
from AutoNN.preprocessing import dataset_container as dc
from AutoNN.preprocessing import date_parsing as dp
from AutoNN.preprocessing import column_info as ci
from AutoNN.preprocessing import encoding as enc
from AutoNN.preprocessing import nan_handling as nanhandle
from AutoNN.preprocessing import feature_elimination as fe

import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataCleaning:

    def __init__(self, label: list(), train_dataframe: dd, validation_dataframe = None, test_dataframe = None, override = False, threshold = 20) -> None:
        self.__dataset = dc.DatasetContainer(label, train_dataframe, validation_dataframe, test_dataframe, override)
        self.__pipeline = None
        self.__regression_threshold = threshold
    
    def parse_dates(self):
        date_parse = dp.DateTime_Parsing(self.__dataset)
        date_parse.parse_dates()

    def generate_column_info(self):
        colinf = ci.ColumnInfo(self.__dataset)
        colinf.generate_info()
        self.__col_info = colinf.column_info

    def clean_data(self, override_imputer = {}):
        self.__nan_handling = nanhandle.DataHandling(dataset=self.__dataset, col_inf=self.__col_info)
        self.__nan_handling.run_cleaner(override_imputer)
        self.__dataset = self.__nan_handling.dataset

    def is_regression(self) -> bool:
        cardinal = self.__col_info(self.__dataset.get_label())['cardinality']
        if cardinal > self.__regression_threshold:
            return [1, -1]
        else:
            return [0, cardinal]

    def get_label(self):
        return self.__dataset.get_label()

    def encode(self, type = "train"):
        # Choose columns to be encoded
        # Send and recieve encoded dataframe
        encoding = enc.Encoding()

        for col_name in self.__col_info:
            if self.__col_info[col_name]["dtype"] == np.dtype("object"):
                encoding.send_column(column = self.__dataset.get(types = [type])[0][col_name], col_name = col_name)

        enc_df = encoding.encode(dataframe = self.__dataset.get(types = [type])[0], enc_type = "label")
        self.__dataset.set(enc_df, type = type)
        print(encoding.key_dict)

    def scaling(self, type = "train"):
        # min max scaling
        scaler_x = MinMaxScaler()
        scaler_x.fit(self.__dataset.get(types = [type])[0])
        scaled_np_arr = scaler_x.transform(self.__dataset.get(types = [type])[0])
        # scaled np array

    def feature_elimination(self, type = "train", method = "correlation", percentage_column_drop = None, override = False):
        feature_elim = fe.FeatureElimination()
        if method == "recursive":
            feature_elim.recursive_feature_elimination_fit(self.__dataset.get(types = [type])[0], self.__col_info, percentage_column_drop, override)
        elif method == "correlation":
            feature_elim.correlation_feature_elimination_fit(self.__dataset.get(types = [type])[0], self.__dataset.get_label()[0], threshold = 0.1)

        eliminated_df = feature_elim.eliminate_features(self.__dataset.get(types = [type])[0], self.__dataset.get_label()[0])
        self.__dataset.set(eliminated_df, type = type)
        self.generate_column_info()

    @property
    def dataset(self):
        return self.__dataset

    @property
    def column_info(self):
        return self.__col_info