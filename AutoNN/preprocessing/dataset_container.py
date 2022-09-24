from dask import dataframe as dd
from dask_ml.model_selection import train_test_split

class DatasetContainer:
    def __init__(self, label: list(), train_dataframe: dd,validation_dataframe = None, test_dataframe = None, override = False) -> None:
        self.__dataset = dict([('train', train_dataframe),('validation', validation_dataframe),('test', test_dataframe)])
        self.__label = label
        self.__override = override
        self.__basic_cleaning()


    def __basic_cleaning(self) -> None:
        types = ['train', 'validation', 'test']
        for type in types:
            try:
                holder = self.__dataset.get([type])[0].dropna(axis = 0, subset = self.get_label())
                holder.reset_index(drop = True, inplace = True)
                self.__dataset.set(type, holder)
            except:
                continue
    

    def __is_override(self) -> bool:
        return self.__override


    def get_columns(self) -> list:
        return list(self.__dataset['train'].columns)


    def get_label(self) -> list:
        return self.__label


    def set(self, dataset: dd, type = 'train'):
        self.__dataset[type] = dataset


    def get(self, types = ['train']) -> dd:
        dataset = list()
        for type in types:
            dataset.append(self.__dataset[type])
        return dataset


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