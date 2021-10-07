from datetime import datetime
import joblib
import numpy as np
import pandas as pd

from src.data.features import generate_features


def training_workload(**kwargs):
    ## Fetch data
    ## Splitting/caching
    ## Preprocessing
    ##  - Valuable to have a preprocessing function that can
    ##      apply the same transformations to the data in both
    ##      training and prediction
    ## Train model
    ## Evaluate
    ## Store model artifacts
    
    return "Great model"


def train_validate_test_split(x, y, test_size, random_state):
    x_train, x_test_temp, y_train, y_test_temp = train_test_split(x,y, test_size = 0.5, random_state=42)
    x_validate, x_test, y_validate, y_test = train_test_split(x_test_temp, y_test_temp, test_size = 0.5, random_state=42)
    return x_train, y_train, x_validate, y_validate, x_test, y_test