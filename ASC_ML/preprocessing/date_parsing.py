from datetime import datetime
from dask import dataframe as dd
import re

class DateTime_Parsing:

    def __init__(self) -> None:
        self.__rows_to_parse = dict()

    def fit_transform(self, dataframe) -> dd:
        self.fit(dataframe)
        return self.transform(dataframe)

    def fit(self, dataframe):
        cols = dataframe.columns
        for col in cols:
            if bool(re.search("\\bdt_(.*)", col)):
                new_col_name = re.findall("\\bdt_(.*)", col)[0]
                if new_col_name in cols:
                    new_col_name = col
                self.__rows_to_parse[col] = new_col_name

    def transform(self, dataframe) -> dd:
        for col in self.__rows_to_parse.keys():
            dataframe[self.__rows_to_parse[col]] = dd.to_datetime(dataframe.pop(col))
        return dataframe