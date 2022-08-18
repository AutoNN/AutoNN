from sklearn.pipeline import Pipeline

class Basic_Data_Cleaning:
    def __init__(self, df_x, df_y):
        self._df_x_clean = df_x
        self._df_y_clean = df_y
        self._pipeline = None
    
    @property
    def df_x_clean(self):
        return self._df_x_clean

    @property
    def df_y_clean(self):
        return self._df_y_clean

    @property
    def pipeline(self):
        return self._pipeline

    def implement_pipeline(self):
        None
    
    def _define_pipeline(self):
        # NaN Removal/Replacement (Imputation)
        # Scaling (?Normalization)
        # Categorical Encoding - Label, One Hot, Ordinal
        # Correlation Wise Column removal
        # ?Outlier Removal (Isolation Forest)
        # 
        None

    