from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import dask.dataframe as dd

class FeatureElimination:
    def __init__(self):
        self._column_names_kept = []

    def recursive_feature_elimination_fit(self, dataframe, col_info, percentage_column_drop = None, override = False):
        # override - To not take into account no of columns and drop 20% columns
        X, Y, Y_cardinalities, no_of_columns = self._get_X_Y_cardinality(dataframe, col_info)

        if Y_cardinalities[0] < 15:
            model = RandomForestClassifier()
        else:
            model = RandomForestRegressor()

        if no_of_columns > 40:
            n = no_of_columns*0.2
        else:
            n = no_of_columns
        
        if override == True:
            n = no_of_columns*0.2
        if percentage_column_drop != None:
            n = no_of_columns * percentage_column_drop

        rfe = RFE(estimator = model, n_features_to_select = n)
        fit = rfe.fit(X, Y)
        self._column_names_kept = X.columns[fit.support_]
        # X = X[self._column_names_kept]
        # total_df = dd.concat([X,Y], axis = 1)

    def correlation_feature_elimination_fit(self, dataframe, label_name, threshold = 0.1):
        correlation_matrix = dataframe.compute().corr()
        for column_name in correlation_matrix[label_name].keys():
            if abs(correlation_matrix[label_name][column_name].item()) > 0.1:
                self._column_names_kept.append(column_name)
    
    def eliminate_features(self, dataframe, label_name):
        return dataframe[self._column_names_kept]

    def _get_X_Y_cardinality(self, dataframe, col_info):
        X_columns = []
        Y_columns = []
        Y_cardinalities = []
        for column_name in col_info:
            if col_info[column_name]["is_label"] == True:
                Y_columns.append(column_name)
                Y_cardinalities.append(col_info[column_name]["cardinality"])
            else:
                X_columns.append(column_name)

        X = dataframe[X_columns]
        Y = dataframe[Y_columns]
        no_of_columns = dataframe.shape[-1]
        return X, Y, Y_cardinalities, no_of_columns
        

