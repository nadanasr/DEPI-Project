# Import Libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import json 
import re 
import pickle
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__, template_folder='templates')

# Load Model
model = load_model('SaveModels/model.h5')

with open('intent_responses.json') as file:
    data = json.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get', methods=['POST'])  # Handle POST request
def get_response():
    user_input = request.form['user']  # Get user input from the form
    input_text = user_input.lower()
    user_input = re.sub(r'\s+', ' ', user_input)  # Remove extra spaces
    user_input = re.sub(r'[^\w\s]', '', user_input)  # Remove punctuation

    # Tokenize and pad the input text
    tokenizer = pickle.load(open('SaveModels/tokenizer.pkl', 'rb'))
    input_sequence = tokenizer.texts_to_sequences([user_input])  # Pass as a list
    input_sequence = pad_sequences(input_sequence, maxlen=20)  # Use a fixed max length

    # Predict the intent using the model
    prediction = model.predict(input_sequence)
    intent_index = np.argmax(prediction)
    intents = list(data.keys())
    predicted_intent = intents[intent_index]

    # Fetch a random response for the predicted intent
    response = random.choice(data[predicted_intent])

    return render_template('index.html', response=response)
    # return jsonify({'response': response})  # Return the response as JSON

if __name__ == "__main__":
    app.run(debug=True)
