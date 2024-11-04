from mlflow.models import validate_serving_input
import json
import random
import numpy as np

model_uri = 'runs:/9df7e90f4644459e8763a9e88e4b2d5f/LSTM_Customer_Support_Model'

# The logged model does not contain an input_example.
# Manually generate a serving payload to verify your model prior to deployment.
from mlflow.models import convert_input_example_to_serving_input

data = {
    "X_test": [
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            21,
            5,
            164,
            34,
            122,
            22,
            18
        ]
    ]
}

with open('intent_responses.json', 'r') as f:
    responses = json.load(f)

# Define INPUT_EXAMPLE via assignment with your own input example to the model
# A valid input example is a data instance suitable for pyfunc prediction
serving_payload = convert_input_example_to_serving_input(data)

# Validate the serving payload works on the model
prediction = validate_serving_input(model_uri, serving_payload)

# Step 7: Get the predicted intent
intent_index = np.argmax(prediction)  # Assuming softmax output
intents = list(responses.keys())
predicted_intent = intents[intent_index]

# Step 8: Fetch a random response for the predicted intent
response = random.choice(responses[predicted_intent])
print(response)