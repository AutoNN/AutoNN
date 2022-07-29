from select import select
from dask import dataframe as dd
from sklearn import pipeline

class DataCleaning:

    def __init__(self, dataframe: dd, label: str) -> None:
        self.__dataset = dataframe
        self.__pipeline = None
        self.__label = label


    def __cardinality(self, col) -> int:
        return self.__dataset[col].nunique()

    
    def __str__(self) -> str:
        return self.__dataset.compute()


    @staticmethod
    def is_regression(self) -> bool:
        cardinal = self.__cardinality(col = self.__label)
        if cardinal > 20:
            return True
        else:
            return False
