import joblib
from processes.text_processor import textProcess
from processes.target_processor import TargetProcessor
from processes.vectorizer import Vectorizer
from processes.model_tester import Tester
from processes.trainer import Trainer
import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin
from analysis.compare_models import compare
from sklearn.model_selection import GridSearchCV
from processes.hyperparameters import models_with_params
import logging

class TrainProcess:

    def __init__(   self, file_path:str, file_type:str, 
                    target_col_name:str, input_col_name:str,
                    model_abbrivation='lr', 
                    vectorizer_abbrivation='count', 
                    MIN_DF=0.001
                            ) -> None:
        self.file_path = file_path
        self.file_type = file_type
        self.target_col_name = target_col_name
        self.input_col_name = input_col_name
        self.model_abbrivation = model_abbrivation
        self.vectorizer_abbrivation = vectorizer_abbrivation
        self.MIN_DF = MIN_DF


    def readCsv(self) -> None:
        """ read csv file and store it in class objects df variable
            Return: None"""
        if self.file_type == '.csv':
            self.df = pd.read_csv(self.file_path)
        elif self.file_type == '.xlsx':
            self.df = pd.read_excel(self.file_path)

    def split(self) -> tuple:
        """ Splits data into X and y
            Return: tuple"""
        self.X = self.df[self.input_col_name]
        self.y = self.df[self.target_col_name]
        return self.X, self.y

    def analyze(
                self, shuffle:bool=True, 
                test_size:float=0.33, 
                avoid_numerical_text:bool=True, 
                all:bool=True, **params
                ) -> None:
        """ 
            process data and analyze it with graphs
            Return: None
            """
        X, y = self.processData(vectorizer_abbrivation=self.vectorizer_abbrivation,
                                MIN_DF=self.MIN_DF, 
                                avoid_numerical_text=avoid_numerical_text)

        tester = Tester(stratify=y, 
                        shuffle=shuffle, 
                        test_size=test_size)
        if all:
            results = tester.testAllModels(X=X, y=y)
        else:
            results = tester.testAModel(X=X, y=y, 
                                        model_abbrivation=self.model_abbrivation ,
                                        **params)
        compare(results=results)

    def testAll(self, vectorizer_abbrivation:str='tf-idf', MIN_DF:float=0.01):
        X, y = self.processData(vectorizer_abbrivation='tf-idf', MIN_DF=MIN_DF)
        tester = Tester()
        tester.testAllModels(X=X, y=y)
        
    def bestClassifier(self, X:pd.DataFrame, y:pd.Series, model:str) -> ClassifierMixin:
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
        grid_search.best_estimator_

    def bestClassifierSelector(self, X:pd.DataFrame, y:pd.Series) -> ClassifierMixin:
        """ Try all possible combinations of parameters and find best version of calssifier
        
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
            joblib.dump(best_estimator, f"train_nlp\\trained_model\\{best_model_name}.pkl")
        return result

    def processData(self, vectorizer_abbrivation:str, MIN_DF:float=0.01, 
                    avoid_numerical_text:bool=True) -> tuple:
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
        print("Processing texts...")
        self.df[self.input_col_name] = self.df[self.input_col_name].apply(lambda x: textProcess(x))
        print("Text processing is done")
        targer_process = TargetProcessor()
        print("Processing target...")
        df = targer_process.labelEncoder(df=self.df, column_name=self.target_col_name)
        print("Target processing is done")
        vectorizer_obj = Vectorizer(MIN_DF=MIN_DF, avoid_numerical_text=avoid_numerical_text)
        print("Vectorizing text ...")
        vectors = vectorizer_obj.vectorize(vectorizer_abbrivation=vectorizer_abbrivation, 
                                           df=df, column_name=self.input_col_name)
        
        return vectors, df[self.target_col_name]
    
    def trainAmodel(self, model_abbrivation:str, 
                    vectorizer_abbrivation:str, MIN_DF:float=0.01, 
                    avoid_numerical_text:bool=True, **params) -> ClassifierMixin:
        
        """ Process data and train a model
            Return: ClassifierMixin"""
        X, y = self.processData(vectorizer_abbrivation=vectorizer_abbrivation, 
                              MIN_DF=MIN_DF, avoid_numerical_text=avoid_numerical_text)
        trainer = Trainer()
        trainer.fit(X_train=X, y_train=y)
        model = trainer.trainAmodel(model_abbrivation=model_abbrivation, **params)
        return model
    
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