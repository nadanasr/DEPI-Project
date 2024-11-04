#Import Liberaries
import numpy as np
from flask import Flask, request, jsonify, render_template
import json 
import re 
from sklearn.preprocessing import LabelEncoder
import numpy as np 
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow.keras.models import load_model
import random


app = Flask(__name__, template_folder='templates')

#load Model
model=load_model('SaveModels/model.h5')

with open('intent_responses.json') as file:
    data=json.load(file)

@app.route('/')
def home():
    return render_template('home.html')



def get_response(user_input):

    input_text = user_input.lower()
    user_input = re.sub(r'\s+', ' ', user_input)  # Remove extra spaces
    user_input = re.sub(r'[^\w\s]', '', user_input)  # Remove punctuation

    # Tokenize and pad the input text
    tokenizer = pickle.load(open(r'SaveModels\tokenizer.pkl'))
    le = pickle.load(open(r'SaveModels\label_encoder.pkl'))
    input_sequence = tokenizer.texts_to_sequences([user_input])  # Pass as a list
    input_sequence = pad_sequences(input_sequence, maxlen=20)  # Use a fixed max length

    # Step 6: Predict the intent using the model
    prediction = model.predict(input_sequence)  # Ensure input is in the right format

    # Step 7: Get the predicted intent
    intent_index = np.argmax(prediction)  # Assuming softmax output
    intents = list(data.keys())
    predicted_intent = intents[intent_index]

    # Step 8: Fetch a random response for the predicted intent
    response = random.choice(data[predicted_intent])

    return render_template('home.html', prediction_text='{} '.format(response))


if __name__ == "__main__":
    app.run(debug=True)
