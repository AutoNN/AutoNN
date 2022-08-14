from sklearn.datasets import load_iris
data = load_iris()

#import dask.dataframe as dd
#df = dd.read_("E:\Seminar Conferences\Cough Based Disease Detection\Cough analysis using ML\Spectral_Feature.csv").persist()

import numpy as np
full_data = np.append(data.data, data.target.reshape(-1,1), axis = 1)

import pandas as pd
df = pd.DataFrame(full_data)
df.columns = data.feature_names + ["Type"]

df.to_csv("iris.csv")

from autoviz.AutoViz_Class import AutoViz_Class
%matplotlib inline
AV = AutoViz_Class()

viz = AV.AutoViz("iris.csv", sep = ",")
