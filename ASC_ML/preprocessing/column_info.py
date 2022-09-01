class ColumnInfo:
    def __init__(self, dataset) -> None:
        self.__dataset = dataset
        self.__columns = dataset.get_columns()
        self.__column_info = dict((column, dict()) for column in self.__columns)


    def __cardinality(self, col: str) -> int:
        return self.__dataset.get(['train'])[0][col].nunique(dropna = True).compute()


    def __percentage_missing(self, col_name) -> int:
        return self.__dataset.get(['train'])[0][col_name].isnull().mean().compute()

    
    def __datatype(self, col_name):
        return self.__dataset.get(['train'])[0][col_name].dtype

    
    def generate_info(self):
        for col in self.__columns:
            self.__column_info[col]['dtype'] = self.__datatype(col)
            self.__column_info[col]['is_label'] = col in self.__dataset.get_label()
            self.__column_info[col]['missing'] = self.__percentage_missing(col)
            self.__column_info[col]['cardinality'] = self.__cardinality(col)
    
    @property
    def column_info(self) -> dict:
        return self.__column_info