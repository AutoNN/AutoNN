from dask import dataframe as dd
from preprocessing import dataset_container as dc
from preprocessing import column_info as ci

class DataCleaning:

    def __init__(self, label: list(), train_dataframe: dd, validation_dataframe = None, test_dataframe = None, override = False, threshold = 20) -> None:
        self.__dataset = dc.DatasetContainer(label, train_dataframe, validation_dataframe, test_dataframe, override)
        colinf = ci.ColumnInfo(self.__dataset)
        colinf.generate_info()
        self.__col_info = colinf.get_info()
        self.__pipeline = None
        self.__regression_threshold = threshold


    def is_regression(self) -> bool:
        cardinal = self.__col_info(self.__dataset.get_label())['cardinality']
        if cardinal > self.__regression_threshold:
            return [1, -1]
        else:
            return [0, cardinal]

    def get_label(self):
        return self.__dataset.get_label()

    @property
    def dataset(self):
        return self.__dataset

    @property
    def col_info(self):
        return self.__col_info