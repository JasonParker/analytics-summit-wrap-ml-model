from datetime import datetime
import joblib
import json
import pandas as pd

from src.data.features import preprocessing


def scoring_workload(**kwargs):
    ## Fetch data
    ## Load model
    ## Preprocessing
    ##  - Valuable to have a preprocessing function that can
    ##      apply the same transformations to the data in both
    ##      training and prediction
    ## Generate predictions
    ## Postprocessing
    ##  - Maybe you want to apply a post hoc rule to the output

    return "Predictions!"