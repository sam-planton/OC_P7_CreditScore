import mlflow.pyfunc
from flask import Flask, jsonify, request
import pandas as pd

# Load the model from the server using its URI
mlflow.set_tracking_uri("http://13.37.31.96:5000")
client = mlflow.tracking.MlflowClient()
experiment = mlflow.get_experiment_by_name('MLflow_FinalModel')
runs = mlflow.search_runs(experiment_ids=experiment.experiment_id)
run_id = \
    runs[runs['tags.mlflow.runName'] == 'LogisticRegression'].run_id.values[0]
run = client.get_run(run_id)
artifacts_uri = run.info.artifact_uri
model_uri = f"{artifacts_uri}/model"
# model = mlflow.pyfunc.load_model(model_uri)
model = mlflow.sklearn.load_model(model_uri)
print(model_uri)

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
