from datetime import datetime
import joblib
import json
import pandas as pd

from src.data.features import generate_features


def scoring_workload(**kwargs):
    print(kwargs.get('data'))
    if kwargs.get('data') is not None:
        data = pd.DataFrame(kwargs.get('data'))
    else: data = pd.read_csv('diabetes_test_x.csv', delimiter = ',')

    msg_data = f"Data dimensions: {data.shape}"

    print(msg_data)

    x_score = generate_features(data)
    print(x_score.head())

    model_package = joblib.load('models/diabetes_predictor')
    pipe = model_package['model_pipeline']
    print(pipe)

    print(x_score.columns)
    y_predict = pipe.predict(x_score)

    data['prediction'] = y_predict  

    return {
        'name': model_package['name'],
        'model_version': model_package['model_version'],
        'model_type': model_package['model_type'],
        'model_objective': model_package['model_objective'],
        'model_algorithm': model_package['model_algorithm'],
        'model_trained_date': model_package['model_trained_date'],
        'data_dimensions': msg_data,
        #'prediction_data': data.to_dict(orient='records'),
        'prediction_data': data.to_dict(orient='records'),
        'scoring_details': {
            'alert_rate': round(data[data['prediction']==1].shape[0]/data.shape[0], 3),
            'alert_count': data[data['prediction']==1].shape[0],
            'prediction_count': data.shape[0],
            'timestamp': str(datetime.utcnow())[0:19]
        }
    }