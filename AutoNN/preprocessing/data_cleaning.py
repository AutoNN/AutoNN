from dask import dataframe as dd
from AutoNN.preprocessing import dataset_container as dc
from AutoNN.preprocessing import date_parsing as dp
from AutoNN.preprocessing import column_info as ci
from AutoNN.preprocessing import encoding as enc
from AutoNN.preprocessing import nan_handling as nanhandle
from AutoNN.preprocessing import feature_elimination as fe

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataCleaning:

    def __init__(self, label: list(), train_dataframe: dd, validation_dataframe = None, test_dataframe = None, override = False, threshold = 20) -> None:
        self.__dataset = dc.DatasetContainer(label, train_dataframe, validation_dataframe, test_dataframe, override)
        self.__pipeline = None
        self.__regression_threshold = threshold
        self.__feature_eliminator = None
        self.__scaler = None
    
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
        cardinal = self.__col_info[self.__dataset.get_label()[0]]['cardinality']
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

    def scaling_fit(self, dataframe):
        # min max scaling
        self.__scaler = MinMaxScaler()
        self.__scaler.fit(dataframe)
        
    def scaling_transform(self, dataframe):
        columns = list(dataframe.columns)
        scaled_dset = pd.DataFrame(self.__scaler.transform(dataframe), columns=columns)
        scaled_dset = dd.from_pandas(scaled_dset, npartitions=1)
        scaled_dset = scaled_dset.repartition(npartitions=1)
        scaled_dset = scaled_dset.reset_index(drop=True)
        return scaled_dset

    def feature_elimination_fit(self, type = "train", method = "correlation", percentage_column_drop = None, override = False):
        self.__feature_eliminator = fe.FeatureElimination()
        if method == "recursive":
            self.__feature_eliminator.recursive_feature_elimination_fit(self.__dataset.get(types = [type])[0], self.__col_info, percentage_column_drop, override)
        elif method == "correlation":
            self.__feature_eliminator.correlation_feature_elimination_fit(self.__dataset.get(types = [type])[0], self.__dataset.get_label()[0], threshold = 0.1)

    def eliminate_features(self, type = "train"):
        eliminated_df = self.__feature_eliminator.eliminate_features(self.__dataset.get(types = [type])[0], self.__dataset.get_label()[0])
        self.__dataset.set(eliminated_df, type = type)
        self.generate_column_info()

    @property
    def dataset(self):
        return self.__dataset

    @property
    def column_info(self):
        return self.__col_info