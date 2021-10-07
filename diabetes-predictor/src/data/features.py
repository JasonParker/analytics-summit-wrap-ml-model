import numpy as np
import pandas as pd


def preprocessing(df):
    result = df.copy(deep=True)
    result['hyperglycemic_flag'] = create_hyperglyemic_flag(result['Glucose'])
    result['hypoglycemic_flag'] = create_hypoglyemic_flag(result['Glucose'])
    return result


def create_hyperglyemic_flag(series, **kwargs):
    return np.where(
        series > (kwargs.get('glucose_value') or 240), 1, 0
    )

def create_hypoglyemic_flag(series, **kwargs):
    return np.where(
        series < (kwargs.get('glucose_value') or 100), 1, 0
    )


