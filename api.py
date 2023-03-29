import mlflow.pyfunc
from flask import Flask, jsonify, request
import pandas as pd

# Import the model
model = mlflow.sklearn.load_model('data/model')

# Initialize Flask app
app = Flask(__name__)


# Define a route for making predictions
@app.route("/predict", methods=["POST"])
def predict():
    # Get the input data from the request body
    data = request.json

    # Convert the input data to a pandas DataFrame
    input_df = pd.DataFrame(data)
    input_df = input_df.astype("float32")

    # Get predictions from the loaded model
    predictions = model.predict_proba(input_df.to_numpy())[:, 1]

    # Return the predictions as a JSON response
    return jsonify(predictions.tolist())


if __name__ == '__main__':
    # Start Flask app
    app.run(debug=True, host='0.0.0.0', port=8000)
