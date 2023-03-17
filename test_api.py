import requests
import pandas as pd
import os

API_URL = os.environ.get("API_URL")

def test_api_prediction():
    # Load the data sample from the CSV file
    data_sample = pd.read_csv("data_sample.csv")

    # Send a POST request to the API endpoint with the data as the request body
    response = requests.post(API_URL, json=data_sample.to_dict())

    # Check that the response status code is 200 OK
    assert response.status_code == 200

    # Check that the response body contains valid JSON data
    assert response.headers["content-type"] == "application/json"

    # Parse the JSON data from the response body
    json_response = response.json()

    # Check that the JSON response contains the expected keys
    expected_keys = ["predictions"]
    assert set(expected_keys).issubset(json_response.keys())

    # Check that the predictions are in the expected format
    predictions = json_response["predictions"]
    assert isinstance(predictions, list)
    assert all(isinstance(prediction, float) for prediction in predictions)