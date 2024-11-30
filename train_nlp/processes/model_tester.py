from processes.trainer import Trainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
import pandas as pd
import numpy as np
from typing import Union
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin


class Tester:

    trainer = Trainer()

    def __init__(self, stratify:Union[pd.Series, np.ndarray]=None, shuffle:bool=True, test_size:float=0.33) -> None:
        self.stratify = stratify
        self.shuffle = shuffle
        self.test_size = test_size

    def splitter(self, X:Union[pd.DataFrame, np.ndarray], y:Union[pd.Series, np.ndarray]):
        """ Split the X and y paramters and return 
            Args: 
                X: pd.DataFrame
                y: pd.Series
                """
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=self.test_size, 
                                                            stratify=self.stratify,
                                                            shuffle=True, random_state=42)
        return (X_train, X_test, y_train, y_test)
    

    def classificationMatrices(self, y_true:pd.Series, y_pred:pd.Series) -> dict:
        """ 
            calculates (accuracy, precision, recall, f1) from given y_true and y_pred

            Args:
                y_true: pd.Series (True target values)
                y_pred: pd.Series (Predicted target values)

            Return: 
                dict with all metrics
            """
        
        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
        recall = recall_score(y_true=y_true, y_pred=y_pred, average=None)
        precision = precision_score(y_true=y_true, y_pred=y_pred, average=None)
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average=None)
        return {'accuracy':accuracy, 'precision':precision, 'recall':recall, 'f1':f1}

    def testAllModels(self, X:Union[pd.DataFrame], y:Union[pd.Series]) -> dict[dict]:
        """ tain every model of Trainer class and evaluate it with below matrics
            train_score, test_score, accuracy_score precision_score, recall_score, f1_score_score
            Args: 
                df (dataframe to work on it)
                column_name (target columns name)
                  
            Return: 
                dictionary with all matrix
            """

        X_train, X_test, y_train, y_test = self.splitter(X=X, y=y)
            
        self.trainer.fit(X_train=X_train, y_train=y_train)
        lr = self.trainer.logisticRegression()
        dt = self.trainer.decisionTreeClassifier()
        nb = self.trainer.multinomialNB()
        rf = self.trainer.randomForestClassifer()

        model_reports = {}
        model_lis = [lr, dt, nb, rf]
        for model in model_lis:
            y_pred = model.predict(X_test)
            report = self.classificationMatrices(y_true=y_test, y_pred=y_pred)
            model_reports[str(model)] = report
        return model_reports

    def testAModel(self, X:Union[pd.DataFrame], y:Union[pd.Series], model_abbrivation:str, **params) -> dict[dict]:
        """ tain every model of Trainer class and evaluate it with below matrics
            train_score, test_score, accuracy_score precision_score, recall_score, f1_score_score
            Args: 
                X: pd.DataFrame (input data for model training)
                y: pd.Series (input data with target values)
                model_abbrivation: str (choose from ('lr', 'dt', 'nb', 'rf'))
                
            Return: 
                dictionary with all matrix"""
        
        X_train, X_test, y_train, y_test = self.splitter(X=X, y=y)
        self.trainer.fit(X_train=X_train, y_train=y_train)

        def predictor(model:ClassifierMixin) -> dict[dict]:
            y_pred = model.predict(X_test)
            return self.classificationMatrices(y_true=y_test, y_pred=y_pred)

        if model_abbrivation == 'lr':
            model = self.trainer.logisticRegression(**params)
            return predictor(model=model)
        
        if model_abbrivation == 'dt':
            model = self.trainer.decisionTreeClassifier(**params)
            return predictor(model=model)
        
        if model_abbrivation == 'nb':
            model = self.trainer.multinomialNB(**params)
            return predictor(model=model)
        
        if model_abbrivation == 'rf':
            model = self.trainer.randomForestClassifer(**params)
            return predictor(model=model)
        
    def tester(self, X:Union[pd.DataFrame], y:Union[pd.Series], model_abbrivation:str, tester_abbrivation:str, **params) -> dict[dict]:
        """ test model or models based on test_abbrivation
            Args:
                """
        if tester_abbrivation == 'one':
            return self.testAModel(X=X, y=y, model_abbrivation=model_abbrivation, **params)
        elif tester_abbrivation == 'all':
            return self.testAllModels(X=X, y=y)

        raise ValueError("wrong tester_abbrivation choose from ('one', 'all') ")