from flask import Flask

import urllib.request
import json
import os
import ssl
# Import the necessary librari
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from flask import Flask,jsonify
from flask import request
from flask_cors import CORS
app = Flask(__name__)
CORS(app, support_credentials=True)
# def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    # if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        # ssl._create_default_https_context = ssl._create_unverified_context

# allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.
# Load the dataset (you can replace this with your dataset)
df = pd.read_csv('./fetal_health.csv')
X = df.drop('fetal_health', axis=1).copy()
y = df['fetal_health'].copy()

y[y == 1] = 0

y[y == 2] = 1
y[y == 3] = 2

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an XGBoost classifier for multiclass classification
model = xgb.XGBClassifier(
    objective='multi:softmax',  # Multiclass classification
    num_class=len(set(y)),      # Number of classes
    max_depth=3,                # Maximum depth of each tree
    n_estimators=100,           # Number of boosting rounds
    learning_rate=0.1,          # Learning rate
    random_state=42             # Random seed for reproducibility
)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.2f}")

# Display classification report
class_report = classification_report(y_test, y_pred)
# print("Classification Report:\n", class_report)

@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/about')
def about():
    return 'About'

@app.route('/classify',methods=["POST"])
def classify():
    my_data=request.get_json()
    feature_columns = [
        'baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions',
        'light_decelerations', 'severe_decelerations', 'prolongued_decelerations',
        'abnormal_short_term_variability', 'mean_value_of_short_term_variability',
        'percentage_of_time_with_abnormal_long_term_variability',
        'mean_value_of_long_term_variability', 'histogram_width', 'histogram_min',
        'histogram_max', 'histogram_number_of_peaks', 'histogram_number_of_zeroes',
        'histogram_mode', 'histogram_mean', 'histogram_median', 'histogram_variance',
        'histogram_tendency'
    ]
    input_data_df = pd.DataFrame(np.array([my_data["data"]]), columns=feature_columns)
    
    # Now, you can use input_data_df for making predictions
    predictions = model.predict(input_data_df)
    # model.predict()
    # print(predictions)
    # print(type(predictions))
    return (predictions.tolist())

if __name__ == '__main__':
   app.run(debug = True)
