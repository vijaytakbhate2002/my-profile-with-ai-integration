import os
import joblib
from processes.text_processor import textProcess
from processes.vectorizer import Vectorizer
from processes.model_tester import Tester
from processes.trainer import Trainer
import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin
from analysis.compare_models import compare
from sklearn.model_selection import GridSearchCV
from processes.hyperparameters import models_with_params
from sklearn.preprocessing import LabelEncoder
from typing import Union
import logging


class PrepareData:
    def __init__(self, file_path:str, file_type:str, target_col_name:str, input_col_name:str) -> None:
        self.file_path = file_path
        self.file_type = file_type
        self.target_col_name = target_col_name
        self.input_col_name = input_col_name

    def readData(self) -> None:
        """ read csv file and store it in class objects df variable
            Return: None"""
        if self.file_type == '.csv':
            self.df = pd.read_csv(self.file_path)
        elif self.file_type == '.xlsx':
            self.df = pd.read_excel(self.file_path)

    def splitData(self) -> tuple:
        """ Splits data into X and y
            Return: tuple"""
        self.X = self.df[self.input_col_name]
        self.y = self.df[self.target_col_name]
        return self.X, self.y



class ProcessData:

    def __init__(self, vectorizer_abbrivation:str, MIN_DF:float=0.01, 
                    avoid_numerical_text:bool=True) -> None:
        
        self.vectorizer_abbrivation=vectorizer_abbrivation
        self.MIN_DF = MIN_DF
        self.avoid_numerical_text =avoid_numerical_text

    def fit(self, X:pd.Series, y:pd.Series) -> None:
        self.X = X
        self.y = y

    def dropNull(self) -> tuple[pd.Series]:
        """ drops null values from X and y pandas series 
            Args:
                X: pd.Series 
                y: pd.Series
                
            Return:
                tuple[pd.Series]"""
        self.X.name = 'input'
        self.y.name = 'output'
        df = pd.concat([X, y], axis='columns')
        df.dropna(inplace=True)
        return df['input'], df['output']
    
    def labelEncoder(self, y:pd.Series) -> Union[pd.Series, pd.DataFrame]:
        """ encode target column with LabelEncoder, replace old labels with new encodings
            
            Args: None
            Return pd.DataFrame
            """
        encoder = LabelEncoder()
        self.y = encoder.fit_transform(y=self.y)
        return self.y
    
    def processData(self) -> tuple:
        """ process text data and vectorize it 
            and return vectorized data attached with processed target column
            
            Args:
                vectorizer_abbrivation: str (choose from ('count', 'tf-idf'))
                MIN_DF: float (words to be neglected from document 
                        eg. 0.01 means word appeard 1% in document will be neglected)
                avoid_numerical_text: bool (if True avoid numbers from text)
                
            Return:
                tuple (X, y)
        """

        X, y = self.dropNull()
        X = X.apply(lambda x: textProcess(x))
        vectorizer_obj = Vectorizer(MIN_DF=self.MIN_DF, avoid_numerical_text=self.avoid_numerical_text)
        vectors = vectorizer_obj.vectorize(X=X, vectorizer_abbrivation=self.vectorizer_abbrivation)
        y = self.labelEncoder(y=y)
        return vectors, y
    
class Analyze:

    def __init__(self, shuffle:bool=True, test_size:float=0.33):
        self.shuffle = shuffle
        self.test_size = test_size


    def analyze(self, all:bool=True, model_abbrivation:str=None, **params) -> None:
        """ 
            process data and analyze it with graphs
            Args:
                all: bool (if True, analysis applies on every algorithm from Tester class)
                model_abbrivation: str (choose from ('lr', 'dt', 'nb', 'rf'))
            Return: 
                dict[dict] (all model metrics)
            """
        
        tester = Tester(stratify=y, 
                        shuffle=self.shuffle, 
                        test_size=self.test_size)
        if all:
            results = tester.testAllModels(X=X, y=y)
        else:
            results = tester.testAModel(X=X, y=y, 
                                        model_abbrivation=model_abbrivation,
                                        **params)
        compare(results=results)
        return results
    

class HyperParameterTuner:

    def __init__(self, model:str, scoring:str) -> None:
        """ Args:
                model: str (choose from ('lr', 'dt', 'rf', 'nb'))
                scoring: str 
                """
        self.model = model
        self.scoring = scoring
        
    def bestEstimator(self, X:pd.DataFrame, y:pd.Series, model:str) -> ClassifierMixin:
        """ Try all possible combinations of parameters and find best version of calssifier
        
            Args:
                X: pd.DataFrame (input data)
                y: pd.Series (target data) 
                model: str (choose from ('lr', 'dt', 'rf', 'nb'))
                
            Return:
                ClassifierMixin
            """

        grid_search = GridSearchCV(estimator=model, param_grid=models_with_params[model][1], 
                                   scoring='f1_weighted', cv=3)
        grid_search.fit(X, y)
        best_estimator = grid_search.best_estimator_
        if os.path.exists(f"train_nlp\\trained_model") == False:
                os.mkdir("train_nlp\\trained_model")    
        joblib.dump(best_estimator, f"train_nlp\\trained_model\\{model}.pkl")

    def bestEstimatorSelector(self, X:pd.DataFrame, y:pd.Series) -> dict[dict]:
        """ Try all possible combinations of parameters on every algorithm
        
            Args:
                X: pd.DataFrame (input data)
                y: pd.Series (target data)
                
            Return:
                dict
            """
        result = {}
        best_score = 0
        best_estimator = None
        best_model_name = None

        for key, val in models_with_params.items():
            print(f"Searching for model {val[2]} ...")
            grid_search = GridSearchCV(estimator=val[0], param_grid=models_with_params[key][1], 
                                    scoring='f1_weighted', cv=3)
            grid_search.fit(X, y)
            result[val[2]] = {
                              'best_estimator':grid_search.best_estimator_, 
                              'best_score':grid_search.best_score_
                              }
            if grid_search.best_score_ > best_score:
                print("best score = ", best_score, "grid searched score = ", grid_search.best_score_)
                best_score = grid_search.best_score_
                best_estimator = grid_search.best_estimator_
                best_model_name = val[2]

        if best_estimator and best_model_name:
            if os.path.exists(f"train_nlp\\trained_model") == False:
                os.mkdir("train_nlp\\trained_model")    
            joblib.dump(best_estimator, f"train_nlp\\trained_model\\{best_model_name}.pkl")

        print("best_model_name", best_model_name)

        return result

class Trainer:
    

    
if __name__ == "__main__":
    process = TrainProcess(
                            file_path="train_nlp\data\question_category.csv", file_type='.csv', 
                            target_col_name='Category', input_col_name='Questions', 
                            model_abbrivation='lr', 
                            vectorizer_abbrivation='count', 
                            MIN_DF=0.001, all=True
                           )
    process.readCsv()
    process.analyze(all=True)

    model =  process.trainAmodel(model_abbrivation='lr', vectorizer_abbrivation='count', 
                        MIN_DF = 0.001, avoid_numerical_text=True)
    print(type(model))
    X, y = process.processData(vectorizer_abbrivation='count',
                               MIN_DF=0.001, avoid_numerical_text=True)
    model = process.bestClassifierSelector(X=X, y=y)
    print(model)