import os
import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# MLflow setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Use your tracking server URL
mlflow.set_experiment("Customer_service_intent_classification")

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

# Model creation
def create_lstm_model(vocab_size, max_sequence_length):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=32, input_length=max_sequence_length))
    model.add(LSTM(64))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(len(label_encoder.classes_), activation="softmax"))
    return model

# Start MLflow run
with mlflow.start_run(run_name = "LSTM"):
    vocab_size = len(tokenizer.word_index) + 1
    
    # Log parameters
    mlflow.log_param("vocab_size", vocab_size)
    mlflow.log_param("max_sequence_length", max_sequence_length)
    mlflow.log_param("lstm_units", 64)
    mlflow.log_param("dense_layer_1_units", 16)
    mlflow.log_param("dense_layer_2_units", 8)
    mlflow.log_param("optimizer", "adam")
    mlflow.log_param("loss_function", "sparse_categorical_crossentropy")

    # Create and compile the model
    model = create_lstm_model(vocab_size, max_sequence_length)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Train the model and log the training metrics
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=64)

    # Log training metrics
    mlflow.log_metric("train_accuracy", history.history['accuracy'][-1])
    mlflow.log_metric("val_accuracy", history.history['val_accuracy'][-1])

    # Evaluate on test data
    loss, acc = model.evaluate(X_test, y_test)
    mlflow.log_metric("test_loss", loss)
    mlflow.log_metric("test_accuracy", acc)

    # Log the model itself to MLflow
    mlflow.keras.log_model(model, "LSTM_Customer_Support_Model")
    
    # Log the tokenizer for future inference use
    tokenizer_path = "tokenizer_lstm.json"
    with open(tokenizer_path, 'w') as f:
        f.write(tokenizer.to_json())
    mlflow.log_artifact(tokenizer_path)

print("LSTM model training and logging completed.")
