import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

class Vectorizer: 
    """ 
        Vectorizatoin process of dataframe with avoid numercal column function 
        MIN_DF: float (tuning parameter of vectorizers to avoid repetitive words from document
                suppose MIN_DF = 0.01 then words with 1% occurance in document will be avoided)

        avoid_numerical_text: bool (True: Drop all possible numerical strings or the strings that can be converted 
                                    into numeric values by int function )        
        """
    
    def __init__(self, MIN_DF:float=0.01, avoid_numerical_text:bool=False) -> None:
        self.MIN_DF = MIN_DF
        self.avoid_numerical_text = avoid_numerical_text

    def avoidNumerical(self, df) -> pd.DataFrame:
        """ Drop all possible numerical strings or the strings that can be converted 
            into numeric values by int function

            Args: df (vectorized dataframe with words as columns names)
            
            Return: pd.Dataframe"""
        
        num_cols = []
        for col in df.columns:
            try:
                col = int(col)
                num_cols.append(col)

            except:
                pass
        df.drop(num_cols, axis='columns', inplace=True)
        return df

    def tf_idfVectorizer(self, X:pd.Series) -> pd.DataFrame:
        """ 
            Apply TF-IDF Vectorizer on given column of dataframe, 
            Check self.avoid_numerical_text if it is True, 
            then call avoidNumerical (Avoids all numerical strings from specified dataframe column)
            
            Args: df (pandas dataframe)
            Return: pd.DataFrame
            """

        tf = TfidfVectorizer(min_df=self.MIN_DF)
        tf_df = tf.fit_transform(X)
        arr = tf_df.toarray()
        df = pd.DataFrame(arr)

        if self.avoid_numerical_text:
            return self.avoidNumerical(df)
        return df
    
    def countVectorizer(self, X:pd.Series) -> pd.DataFrame:
        """ 
            Apply TF-IDF Vectorizer on given column of dataframe, 
            Check self.avoid_numerical_text if itArgs: 
                df: pd.DataFrame (dataframe to apply vectorization)
                colunm_name: str (column on which vectorization should apply)
            Return: pd.DataFrame is True, 
            then call avoidNumerical (Avoids all numerical strings from specified dataframe column)
            
            
            """

        tf = CountVectorizer(min_df=self.MIN_DF)
        tf_df = tf.fit_transform(X)
        arr = tf_df.toarray()
        df = pd.DataFrame(arr)

        if self.avoid_numerical_text:
            return self.avoidNumerical(df)
        return df
    
    def vectorize(self, X:pd.Series, vectorizer_abbrivation:str) -> pd.DataFrame:
        """ apply vectorization on given df and return
            Args:
                vectorizer_abbrivation: str (choose from ('tf-idf', 'count'))
                df: pd.DataFrame (dataframe to apply vectorization)
                colunm_name: str (column on which vectorization should apply)
            Return: pd.DataFrame
            """
        
        if vectorizer_abbrivation == 'tf-idf':
            return self.tf_idfVectorizer(X=X)
        elif vectorizer_abbrivation == 'count':
            return self.countVectorizer(X=X)
        
        raise ValueError("wrong vectorizer_abbrivation choose from ('tf-idf', 'count')")