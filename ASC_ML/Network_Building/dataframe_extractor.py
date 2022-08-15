import os
import dask.dataframe as dd


#Takes path of directory and finds train.csv, test.csv or direct path of singular .csv file
#and returns list of dask dataframe [singular_df_x, singular_df_y] or [train_df_x, train_df_y, test_df_x, test_df_y]
class DataframeExtractor_csv:
    def __init__(self, directory_path, label_names = []):
        self._directory_path = directory_path
        self._label_names = label_names
        self._df_list = []
        self.get_df_list()
        
    @property
    def directory_path(self):
        return self._directory_path
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def df_list(self):
        return self._df_list
    
    #returns list of dask dataframes
    def get_df_list(self):
        
        #If csv file path has been entered
        if self._directory_path.endswith(".csv"):
            print(f"Reading single csv from {self._directory_path}")
            csv_df = dd.read_csv(self._directory_path, assume_missing = True, sample_rows=1000, delimiter=',')
            csv_df.head()
            self._df_list.append(csv_df.loc[:, ~csv_df.columns.isin(self._label_names)])
            self._df_list.append(csv_df[self._label_names])
            
        elif self._no_of_csv(self._directory_path) == 1:
            csv_dir = self._get_csv_path(self._directory_path)
            self._check_dir_exists(csv_dir)
            csv_df = dd.read_csv(csv_dir, assume_missing = True, sample_rows=1000)
            csv_df.head()
            self._df_list.append(csv_df.loc[:, ~csv_df.columns.isin(self._label_names)])
            self._df_list.append(csv_df[self._label_names])
            
        #Finding train.csv and test.csv from directory
        else:            
            print(f"Reading train.csv and test.csv of directory {self._directory_path}")
            
            train_dir = os.path.join(self._directory_path, "train.csv")
            self._check_dir_exists(train_dir)
            csv_df = dd.read_csv(train_dir, assume_missing = True, sample_rows=1000)
            self._df_list.append(csv_df.loc[:, ~csv_df.columns.isin(self._label_names)])
            self._df_list.append(csv_df[self._label_names])
            
            
            test_dir = os.path.join(self._directory_path, "test.csv")
            self._check_dir_exists(test_dir)
            csv_df = dd.read_csv(test_dir, assume_missing = True, sample_rows=1000)
            self._df_list.append(csv_df.loc[:, ~csv_df.columns.isin(self._label_names)])
            self._df_list.append(csv_df[self._label_names])

        
        #If no datasets are found
        if not self._df_list:
            raise EmptyListError("No Datasets found")            
        
    @staticmethod
    def _check_dir_exists(directory):
        if(not os.path.isfile(directory)):
            raise FileNotFoundError(f"Directory {directory} does not exist")
            
    @staticmethod
    def _no_of_csv(directory):
        i = 0
        filenames = os.listdir(directory)
        for filename in filenames:
            if filename.endswith(".csv"):
                i = i+1
        return i

    @staticmethod
    def _get_csv_path(directory):
        filenames = os.listdir(directory)
        for filename in filenames:
            if filename.endswith(".csv"):
                return os.path.join(directory,filename)