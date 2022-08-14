from dask import dataframe as dd
from sklearn.model_selection import train_test_split
from column_info import ColumnInfo

class DatasetContainer:
    def __init__(self, label: str, train_dataframe: dd,validation_dataframe = None, test_dataframe = None, override = False) -> None:
        self.__dataset = dict()
        self.__dataset['train'] = train_dataframe
        self.__dataset['validation'] = validation_dataframe
        self.__dataset['test'] = test_dataframe
        self.__label = label
        self.__override = override

    def __is_override(self) -> bool:
        return self.__override

    def get_columns(self, types = ['train']):
        dataset = list()
        for type in types:
            dataset.append(self.__dataset[type])
        return dataset

    def get_label(self) -> str:
        return self.__label

    def set(self, dataset: dd, type = 'train'):
        self.__dataset[type] = dataset

    def get(self) -> dd:
        return self.__dataset


    def train_test_split(self, test_split = 0.2, validation_split = None, shuffle = True, random_state = None) -> None:
        if self.__is_override or self.__dataset['test'] == None:
            self.__dataset['train'], self.__dataset['test'] = train_test_split(self.__dataset['train'], test_size=test_split, shuffle=shuffle, random_state=random_state)
        else:
            print("Test Dataset already exists")
            print("No changes are done to the Test Dataset")

        if validation_split != None:
            self.train_validation_split(validation_split*((1-test_split)**-1), shuffle, random_state)

    def train_validation_split(self, validation_split = 0.1, shuffle = True, random_state = None) -> None:
        if self.__is_override or self.__dataset['validation'] == None:
            self.__dataset['train'], self.__dataset['validation'] = train_test_split(self.__dataset['train'], test_size=validation_split, shuffle=shuffle, random_state=random_state)
        else:
            print("Validation Dataset already exists")
            print("No changes are done to the Validation Dataset")