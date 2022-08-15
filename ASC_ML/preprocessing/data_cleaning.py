from dask import dataframe as dd
from sklearn import pipeline
from dataset_container import DatasetContainer
from column_info import ColumnInfo

class DataCleaning:

    def __init__(self, label: str, train_dataframe: dd, validation_dataframe = None, test_dataframe = None, override = False, threshold = 20) -> None:
        self.__dataset = DatasetContainer(label, train_dataframe, validation_dataframe, test_dataframe, override)
        self.__col_info = ColumnInfo(self.__dataset)
        self.__pipeline = None
        self.__regression_threshold = threshold


    def is_regression(self) -> bool:
        cardinal = self.__col_info(self.__dataset.get_label())['cardinality']
        if cardinal > self.__regression_threshold:
            return [1, -1]
        else:
            return [0, cardinal]