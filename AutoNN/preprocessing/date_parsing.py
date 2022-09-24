from dask import dataframe as dd
import re

class DateTime_Parsing:

    def __init__(self, dataset) -> None:
        self.__rows_to_parse = dict()
        self.__dset_container = dataset

    def fit_transform(self, dataframe) -> dd:
        self.fit(dataframe)
        return self.transform(dataframe)

    def fit(self, dataframe) -> None:
        cols = dataframe.columns
        for col in cols:
            if bool(re.search("\\bdt_(.*)", col)):
                new_col_name = re.findall("\\bdt_(.*)", col)[0]
                if new_col_name in cols:
                    new_col_name = col
                self.__rows_to_parse[col] = new_col_name

    def transform(self, dataframe) -> dd:
        for col in self.__rows_to_parse.keys():
            new_col = dd.to_datetime(dataframe.pop(col))
            dataframe[self.__rows_to_parse[col]+'_timestamp'] = dd.to_numeric(new_col)//10**9
        return dataframe

    def parse_dates(self) -> None:
        self.fit(self.__dset_container.get()[0])
        types = ['train', 'validation', 'test']
        for type in types:
            if self.__dset_container.get([type])[0] is not None:
                self.__dset_container.set(self.transform(self.__dset_container.get([type])[0]), type)