import os
import json
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Load datasets
Train_PATH = os.path.join(os.getcwd(), 'Training_Dataset.csv')
df_train = pd.read_csv(Train_PATH)

Val_PATH = os.path.join(os.getcwd(), 'Validation_Dataset.csv')
df_val = pd.read_csv(Val_PATH)

Test_PATH = os.path.join(os.getcwd(), 'Testing_Dataset.csv')
df_test = pd.read_csv(Test_PATH)

# Preprocess data
train_texts = df_train['utterance'].tolist()
val_texts = df_val['utterance'].tolist()
test_texts = df_test['utterance'].tolist()

train_labels = df_train['intent'].tolist()
val_labels = df_val['intent'].tolist()
test_labels = df_test['intent'].tolist()

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_labels)
y_val = label_encoder.transform(val_labels)
y_test = label_encoder.transform(test_labels)

# Tokenize the text data
tokenizer = Tokenizer(num_words=5000)  # Keep the top 5000 words
tokenizer.fit_on_texts(train_texts)

train_sequences = tokenizer.texts_to_sequences(train_texts)
val_sequences = tokenizer.texts_to_sequences(val_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

# Padding sequences to ensure they are of equal length
max_sequence_length = max([len(seq) for seq in train_sequences])
X_train = pad_sequences(train_sequences, maxlen=max_sequence_length)
X_val = pad_sequences(val_sequences, maxlen=max_sequence_length)
X_test = pad_sequences(test_sequences, maxlen=max_sequence_length)

data = {
    "X_test": X_test[:1].tolist()
}

# Save the dictionary as a JSON file
with open("test_data.json", "w") as json_file:
    json.dump(data, json_file, indent=4)

print("X_test and y_test have been saved to test_data.json")