from dask import dataframe as dd
from sklearn.model_selection import train_test_split
from column_info import ColumnInfo

class DatasetContainer:
    def __init__(self, label: str, train_dataframe: dd,validation_dataframe = None, test_dataframe = None, override = False) -> None:
        self.__train_dataset = train_dataframe
        self.__validation_dataset = validation_dataframe
        self.__test_dataset = test_dataframe
        self.__label = label
        self.__override = override

    def __is_override(self) -> bool:
        return self.__override

    def get_columns(self):
        return self.__train_dataset.columns

    def get_label(self) -> str:
        return self.__label


    def get(self) -> dd:
        return self.__train_dataset, self.__validation_dataset, self.__test_dataset


    def train_test_split(self, test_split = 0.2, validation_split = None, shuffle = True, random_state = None) -> None:
        if self.__is_override or self.__test_dataset == None:
            self.__train_dataset, self.__test_dataset = train_test_split(self.__train_dataset, test_size=test_split, shuffle=shuffle, random_state=random_state)
        else:
            print("Test Dataset already exists")
            print("No changes are done to the Test Dataset")

        if validation_split != None:
            self.train_validation_split(validation_split*((1-test_split)**-1), shuffle, random_state)

    def train_validation_split(self, validation_split = 0.1, shuffle = True, random_state = None) -> None:
        if self.__is_override or self.__validation_dataset == None:
            self.__train_dataset, self.__validation_dataset = train_test_split(self.__train_dataset, test_size=validation_split, shuffle=shuffle, random_state=random_state)
        else:
            print("Validation Dataset already exists")
            print("No changes are done to the Validation Dataset")