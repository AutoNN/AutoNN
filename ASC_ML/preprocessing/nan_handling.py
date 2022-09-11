from ASC_ML.preprocessing.dataset_container import DatasetContainer as dc
from sklearn.impute import SimpleImputer, KNNImputer

class DataHandling:
    def __init__(self, dataset: dc, col_inf: dict, missing_threshold = 0.2) -> None:
        self.__bucket = dataset
        self.__column_info = col_inf
        self.__dropped = list()
        self.__full = list()
        self.__unfilled = list()
        self.__label = list()
        self.__threshold = missing_threshold
        self.__determine_type()
        self.__allocate()

    def __determine_type(self):
        for col in self.__column_info.keys():
            if self.__column_info[col]['is_label']:
                self.__label.append(col)
            else:
                if self.__column_info[col]['missing'] == 0:
                    self.__full.append(col)
                elif self.__column_info[col]['missing'] > self.__threshold:
                    self.__dropped.append(col)
                else:
                    self.__unfilled.append(col)

    def __allocate(self):
        fill_in_order = list()
        for col in self.__unfilled:
            fill_in_order.append((self.__column_info[col]['missing'],col))
        fill_in_order.sort()
        print(fill_in_order)