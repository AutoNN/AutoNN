from dask import dataframe as dd
from sklearn import pipeline
from sklearn.model_selection import train_test_split

class DataCleaning:

    def __init__(self, label: str, train_dataframe: dd, validation_dataframe = None, test_dataframe = None, override = True) -> None:
        self.__train_dataset = train_dataframe
        self.__validation_dataset = validation_dataframe
        self.__test_dataset = test_dataframe
        self.__pipeline = None
        self.__label = label
        self.__override = override

    
    def __is_override(self) -> bool:
        return self.__override


    def __cardinality(self, col: str) -> int:
        return self.__train_dataset[col].nunique()

    
    def __str__(self) -> str:
        return self.__train_dataset.compute()

    def get(self) -> dd:
        return self.__train_dataset, self.__validation_dataset, self.__test_dataset

    def is_regression(self) -> bool:
        cardinal = self.__cardinality(col = self.__label)
        if cardinal > 20:
            return True
        else:
            return False

    def train_test_split(self, test_split = 0.2, validation_split = None, shuffle = True, random_state = None) -> None:
        if self.__is_override:
            self.__train_dataset, self.__test_dataset = train_test_split(test_size=test_split, shuffle=shuffle, random_state=random_state)
        elif self.__test_dataset == None:
            self.__train_dataset, self.__test_dataset = train_test_split(test_size=test_split, shuffle=shuffle, random_state=random_state)
        else:
            pass

        if validation_split != None:
            self.train_validation_split(validation_split*0.125, shuffle, random_state)

    def train_validation_split(self, validation_split = 0.1, shuffle = True, random_state = None) -> None:
        if self.__is_override:
            self.__train_dataset, self.__validation_dataset = train_test_split(test_size=validation_split, shuffle=shuffle, random_state=random_state)
        elif self.__validation_dataset == None:
            self.__train_dataset, self.__validation_dataset = train_test_split(test_size=validation_split, shuffle=shuffle, random_state=random_state)
        else:
            pass