from sklearn.preprocessing import LabelEncoder
import pandas as pd


class TargetProcessor:

    def __init__(self) -> None:
        pass

    def labelEncoder(self, df:pd.DataFrame, column_name:str) -> pd.DataFrame:
        """ encode target column with LabelEncoder, replace old labels with new encodings
            
            Args: None
            Return pd.DataFrame
            """
        encoder = LabelEncoder()
        df[column_name] = encoder.fit_transform(df[column_name])
        return df
        

