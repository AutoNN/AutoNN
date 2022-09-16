from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import dask.dataframe as dd

class FeatureElimination:
    def __init__(self, dataframe, col_info, percentage_column_drop = None, override = False):
        self._dataframe = dataframe.copy()
        self._col_info = col_info
        self._percentage_column_drop = percentage_column_drop
        # To not take into account no of columns and drop 20% columns
        self._override = override

    def recursive_feature_elimination(self):
        X, Y, Y_cardinalities, no_of_columns = self._get_X_Y_cardinality()

        if Y_cardinalities[0] < 15:
            model = RandomForestClassifier()
        else:
            model = RandomForestRegressor()

        if no_of_columns > 40:
            n = no_of_columns*0.2
        else:
            n = no_of_columns
        
        if self._override == True:
            n = no_of_columns*0.2
        if self._percentage_column_drop != None:
            n = no_of_columns * self._percentage_column_drop

        rfe = RFE(estimator = model, n_features_to_select = n)
        fit = rfe.fit(X, Y)
        X = X[X.columns[fit.support_]]
        total_df = dd.concat([X,Y], axis = 1)
        
        return total_df, fit.support_

    def _get_X_Y_cardinality(self):
        X_columns = []
        Y_columns = []
        Y_cardinalities = []
        for column_name in self._col_info:
            if self._col_info[column_name]["is_label"] == True:
                Y_columns.append(column_name)
                Y_cardinalities.append(self._col_info[column_name]["cardinality"])
            else:
                X_columns.append(column_name)

        X = self._dataframe[X_columns]
        Y = self._dataframe[Y_columns]
        no_of_columns = self._dataframe.shape[-1]
        return X, Y, Y_cardinalities, no_of_columns
