from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.svm import SVC

from src.data.features import generate_features


def training_workload(**kwargs):
    data = pd.read_csv('data/diabetes.csv', delimiter=',')
    x_train, y_train, x_validate, y_validate, x_test, y_test = train_validate_test_split(
        x = data.drop(columns = 'Outcome'), 
        y = data['Outcome'], 
        test_size = .5, 
        random_state = 42
    )

    ## Caching the test data to run against the scoring service
    x_test.to_csv('diabetes_test_x.csv', index=False)
    y_test.to_csv('diabetes_test_y.csv', index=False)

    msg_data_split = f"""

Data splitting dimensions

Training data: {x_train.shape}
Validation data: {x_validate.shape}
Test data: {x_test.shape}
    
    """
    print(msg_data_split)

    x_train = generate_features(x_train)

    pipe = Pipeline(
        [
            ('scaler', StandardScaler()), 
            ('svc', SVC())
        ]
    )

    pipe.fit(x_train, y_train)

    x_validate = generate_features(x_validate)
    y_predict = pipe.predict(x_validate)

    cm = np.array(confusion_matrix(y_validate, y_predict, labels=[0,1]))

    confusion = pd.DataFrame(cm, index=['Not Diabetic', 'Diabetic'], columns=['Predicted Healthy', 'Predicted Diabetes'])

    msg_evaluation = f"""
    
Model evaluation on validation set

Confusion matrix
{confusion}

Classification report
{classification_report(y_validate, y_predict)}
    
    """
    print(msg_evaluation)

    model_package = {
        'model_pipeline': pipe,
        'data_features': x_train.columns.tolist(),
        'name': f'Diabetes predictor',
        'model_version': '0.0',
        'model_type': 'Supervised',
        'model_objective': 'Classification',
        'model_algorithm': 'Support Vector Classification (SVC)',
        'model_trained_date': str(datetime.today().date())
    }
    joblib.dump(model_package, 'models/diabetes_predictor')
    return {
        'name': model_package['name'],
        'model_version': model_package['model_version'],
        'model_type': model_package['model_type'],
        'model_objective': model_package['model_objective'],
        'model_algorithm': model_package['model_algorithm'],
        'model_trained_date': model_package['model_trained_date'],
        'data_splitting': msg_data_split,
        'model_evaluation': msg_evaluation
    }


def train_validate_test_split(x, y, test_size, random_state):
    x_train, x_test_temp, y_train, y_test_temp = train_test_split(x,y, test_size = 0.5, random_state=42)
    x_validate, x_test, y_validate, y_test = train_test_split(x_test_temp, y_test_temp, test_size = 0.5, random_state=42)
    return x_train, y_train, x_validate, y_validate, x_test, y_test


