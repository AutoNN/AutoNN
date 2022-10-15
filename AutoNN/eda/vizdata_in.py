#importing and setting up
import numpy as np
import hvplot.dask
import seaborn as sns
import matplotlib.pyplot as plt
import random as rnd
import dask.dataframe as dd

import re

hvplot.extension('plotly')

class viznn:
    def __init__(self, label: list(), dataset_container) -> None:
        self._traindf = dataset_container.get(types = ['train'])[0]
        
    def findWholeWord(self, w):
        return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

    def viznn(self):
        df_original = self._traindf

        #specify target label
        target_label = 21
        self._traindf.head()  #check dataframe loaded


        # create a generalisation for the data needed for the plotting, by changing the labels

        #1 set the column length count
        length = len(self._traindf.columns)
        original_labels = self._traindf.columns
        #2 rename labels of features to generalise them
        generated_labels = ["Label#{0}".format(i) for i in range(0, length)]
        self._traindf = self._traindf.rename(columns=dict(zip(self._traindf.columns, generated_labels)))
        self._traindf.head() #check renamed column 


        #creating non-numeric column identifiers-------------------------------------------------------------------------
        non_numeric_col_pos = np.empty(length, dtype=object)
        non_numeric_col_pos_counter = 0


        #displays boxplot for all columns to identify outliers and performs segregation of numeric and non-numeric columns----------------------------------------------------------
        for i in range(0, length):
            box_label = "Label#{0}".format(i) #i = column of interest postion (first column starting from 0)
            try:
                box_features = np.array(self._traindf[box_label]).astype('float64')
                fig = plt.figure(figsize =(10, 7))
                plt.boxplot(box_features)

                # Adding title and axes
                current_label = original_labels[i]
                plt.title("Box plot for "+ current_label)
                plt.xlabel('Plot')
                plt.ylabel(current_label)
                plt.show()

            except Exception as e:
                if findWholeWord('could not convert string to float')(str(e)):
                    non_numeric_col_pos[non_numeric_col_pos_counter] = i   #array containing non numeric column positions
                    non_numeric_col_pos_counter = non_numeric_col_pos_counter + 1   



        #plots the correlation matrix for the data -----------------------------------------------------------------------
        heatmap = np.empty((length - non_numeric_col_pos_counter), dtype=object)
        #separate numeric column labels for heatmap plot axes labeling
        j = np.arange(length)
        numeric_counter = 0
        for i in range(0, length):
            if j[i] in non_numeric_col_pos:
                None
            else:
                heatmap[numeric_counter] = original_labels[i]
                numeric_counter = numeric_counter + 1

        sns.set(rc = {'figure.figsize':(15,8)})
        ax = sns.heatmap(self._traindf.corr(), annot=True, xticklabels=list(heatmap), yticklabels=list(heatmap), annot_kws={"size": 35 / np.sqrt(len(self._traindf.corr()))})


        #scatterplot with respect to target variable
        for i in range(0, length):
            if i in non_numeric_col_pos:
                None
            else:
                scatter_target_label = "Label#{0}".format(target_label) #target_label = column number of target_label
                scatter_target_features = np.array(self._traindf[scatter_target_label]).astype('float64')
                scatter_current_target_label = original_labels[target_label]
                if i != target_label:
                    scatter_label = "Label#{0}".format(i) #i = column of interest postion (first column starting from 0)
                    scatter_features = np.array(self._traindf[scatter_label]).astype('float64')
                    current_label = original_labels[i]
                    plt.scatter(scatter_target_features, scatter_features, c ="blue")   
                    plt.xlabel(scatter_current_target_label)
                    plt.ylabel(current_label)
                    plt.show()
