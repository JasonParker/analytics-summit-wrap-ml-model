from flask import Flask, redirect, request
from functools import wraps
import os

from src.modeling.scoring import scoring_workload
from src.modeling.training import training_workload


application = Flask(__name__)

def check_token(view_function):
    @wraps(view_function)
    def secure_route_token_check(*args, **kwargs):
        token = request.args.get('token')
        if token == os.environ['TOKEN']:
            return view_function(*args, **kwargs)
        else:
            return redirect('/')
    return secure_route_token_check


@application.route('/ping')
def home_page():
    """Health check route to ensure app is running."""
    return "pong", 201, {'Content-Type': 'text/html'}


@application.route('/train', methods = ['GET'])
def train_model():
    result = training_workload()
    return result, 201, {'Content-Type': 'text/html'}


@application.route('/score', methods = ['GET', 'POST'])
def score_model():
    print(request.__dict__)
    if request.method == 'GET':
        result = scoring_workload()
    elif request.method == 'POST':
        print(request.get_json())
        result = scoring_workload(data=request.get_json())
    return result, 201, {'Content-Type': 'application/json'}


if __name__ == "__main__":
    application.run()