from datetime import datetime
import joblib
import json
import pandas as pd

from src.data.features import generate_features


def scoring_workload(**kwargs):
    ## Fetch data
    ## Load model
    ## Preprocessing
    ##  - Valuable to have a preprocessing function that can
    ##      apply the same transformations to the data in both
    ##      training and prediction
    ## Generate predictions
    ## Postprocessing

    return "Predictions!"


def postprocessing(data):
    result = """Recommend having a wrapper function that 
    calls more modular functions to execute postprocessing steps. """
    return result 