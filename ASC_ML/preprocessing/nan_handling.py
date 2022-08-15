from dataset_container import DatasetContainer as dc

class missing_data_management:
    def __init__(self, dataset: dc) -> None:
        self.__dataset_bucket = dataset

    
    def remove_row_nan(self):
        types = ['train','valid','test']
        for type in types:
            self.__dataset_bucket.set(self.__dataset_bucket.get(types = [type]).dropna(how='all', subset=self.__dataset_bucket.get_label()))

