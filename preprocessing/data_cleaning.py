from select import select
from dask import dataframe as dd
from sklearn import pipeline

class DataCleaning:

    def __init__(self, dataframe: dd, label: str) -> None:
        self.__dataset = dataframe
        self.__label = label


    def __ordinality(self, col) -> int:
        return self.__dataset[col].nunique()

    
    def __str__(self) -> str:
        return self.__dataset.compute()
