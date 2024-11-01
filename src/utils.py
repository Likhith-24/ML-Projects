import sys
import os

import numpy as np
import pandas as pd
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    try:
        report = {}

        for i in range(len(list(models))): # Looping through the models
            
            model = list(models.values())[i] # Getting the model object
            
            para=params[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)

            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            
            model.fit(X_train,y_train) # Fitting the model
            
            y_train_pred = model.predict(X_train) # Predicting the values for training data
            
            y_test_pred = model.predict(X_test)     # Predicting the values for testing data

            train_model_score = r2_score(y_train,y_train_pred)  # Getting the R2 score for training data

            test_model_score = r2_score(y_test,y_test_pred)    # Getting the R2 score for testing data

            report[list(models.keys())[i]] = test_model_score  # Saving the R2 score in the report dictionary

        return report
    
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)