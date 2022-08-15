class ColumnInfo:
    def __init__(self, dataset) -> None:
        self.__dataset = dataset
        self.__columns = dataset.get_columns()


    def __cardinality(self, col: str) -> int:
        return self.__train_dataset[col].nunique()