from unicodedata import name
from AutoNN.preprocessing.dataset_container import DatasetContainer as dc
from dask_ml.impute import SimpleImputer
from sklearn.impute import KNNImputer
from dask_ml.preprocessing import LabelEncoder
import numpy as np
from dask import dataframe as dd
import pandas as pd

class DataHandling:
    def __init__(self, dataset: dc, col_inf: dict, missing_threshold = 0.2) -> None:
        self.__bucket = dataset
        self.__column_info = col_inf
        self.__dropped = list()
        self.__full = list()
        self.__unfilled = list()
        self.__label = list()
        self.__imputer = dict()
        self.__threshold = missing_threshold
        self.__label_encoder = dict()
        self.__determine_type()
        self.__drop_unwanted()

    def run_cleaner(self, override_imputer = {}):
        self.__allocate(override_imputer)

    def __drop_unwanted(self):
        train, validation, test = self.__bucket.get(['train', 'validation', 'test'])
        self.__bucket.set(dataset=train.drop(self.__dropped, axis=1), type='train')
        if validation is not None:
            self.__bucket.set(dataset=validation.drop(self.__dropped, axis=1), type='validation')
        if test is not None:
            self.__bucket.set(dataset=test.drop(self.__dropped, axis=1), type='test')
    
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
    
    def __imputeKNN(self, dset, colname, imputer):
        columns = self.__full + [colname]
        df_dset = dset[columns].copy()
        dset = dset.drop(columns, axis = 1)
        imputed_nparr = imputer.transform(df_dset)
        # print(f"Imputed SHAPE: {imputed_nparr.shape}")
        # print(f"DF DSET SHAPE: {df_dset.shape} \n")
        if imputed_nparr.shape[-1] == df_dset.shape[-1]:
            zero_arr = np.zeros([imputed_nparr.shape[0],1])
            imputed_nparr = np.append(imputed_nparr, zero_arr, axis = 1)
            # print(imputed_nparr)
        imputed_dset = pd.DataFrame(imputed_nparr, columns=columns+[colname+'_indicator'])

        imputed_dset = dd.from_pandas(imputed_dset, npartitions=1)
        imputed_dset = imputed_dset.repartition(npartitions=1)
        imputed_dset = imputed_dset.reset_index(drop=True)
        dset = dset.repartition(npartitions=1)
        dset = dset.reset_index(drop=True)
        # print(imputed_dset.compute().shape)
        # print(dset.compute().shape)
        dset = dd.multi.concat([dset,imputed_dset], axis = 1, interleave_partitions=True, ignore_unknown_divisions=True, ignore_order = True)
        return dset
    

    def __allocate(self, override = {}):
        fill_in_order = list()
        imputerlist = {'simple':SimpleImputer, 'KNN': KNNImputer}
        for col in self.__unfilled:
            fill_in_order.append((self.__column_info[col]['missing'],col))
        fill_in_order.sort()
        print(fill_in_order)
        for _, colname in fill_in_order:
            selected_imputer = override.get(colname, 'KNN')
            imputer = imputerlist[selected_imputer]
            if selected_imputer == 'simple':
                pass
                # train, valid, test = self.__bucket.get(['train','validation', 'test'])
                # column = np.asarray(train[colname])
                # column = column.reshape(-1,1)
                # new_col_ind = np.isnan(column)
                # imputer.fit(column)
                # new_col = imputer.transform(column)
                # df = dd.DataFrame([new_col, new_col_ind], columns = [colname, colname+"indicator"])
                # print(df)
            elif selected_imputer == 'KNN':
                train, validation, test = self.__bucket.get(['train', 'validation', 'test'])
                columns = self.__full + [colname]
                df_train = train[columns].copy()
                if self.__column_info[colname]['dtype'] == 'object':
                    loaded_imputer = imputer(n_neighbors = 1, add_indicator = True)
                else:
                    loaded_imputer = imputer(n_neighbors = 5, add_indicator = True)
                # print(colname)
                # print(columns)
                # print(len(columns))
                # print(df_train.compute().shape)
                loaded_imputer.fit(df_train)
                self.__bucket.set(dataset = self.__imputeKNN(train, colname, loaded_imputer), type='train')
                if validation is not None:
                    self.__bucket.set(dataset = self.__imputeKNN(validation, colname, loaded_imputer), type='validation')
                if test is not None:
                    self.__bucket.set(dataset = self.__imputeKNN(test, colname, loaded_imputer), type='test')
                self.__imputer.update({colname:imputer})
                # self.__full.append(colname)
                # self.__unfilled.remove(colname)

    @property
    def dataset(self):
        return self.__bucket


            
        