import os
import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Ensure you have MLflow installed: pip install mlflow

# MLflow setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Use your tracking server URL
mlflow.set_experiment("Customer_service_intent_classification")

# Define paths for datasets
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
def create_rnn_model(vocab_size, max_sequence_length):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=50, input_length=max_sequence_length))
    model.add(SimpleRNN(128))
    model.add(Dense(27, activation='softmax'))  # 27 intents
    return model

# Start MLflow run
with mlflow.start_run(run_name = "RNN"):
    vocab_size = len(tokenizer.word_index) + 1
    
    # Log parameters
    mlflow.log_param("vocab_size", vocab_size)
    mlflow.log_param("max_sequence_length", max_sequence_length)
    mlflow.log_param("rnn_units", 128)
    mlflow.log_param("optimizer", "adam")
    mlflow.log_param("loss_function", "sparse_categorical_crossentropy")

    # Create and compile the model
    model = create_rnn_model(vocab_size, max_sequence_length)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Train the model and log the training metrics
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=64)

    # Log training metrics
    mlflow.log_metric("train_accuracy", history.history['accuracy'][-1])
    mlflow.log_metric("val_accuracy", history.history['val_accuracy'][-1])

    # Evaluate on test data
    test_loss, test_acc = model.evaluate(X_test, y_test)
    mlflow.log_metric("test_accuracy", test_acc)

    # Log the model itself to MLflow
    mlflow.keras.log_model(model, "RNN_Customer_Support_Model")
    
    # Log the tokenizer for future inference use
    tokenizer_path = "tokenizer.json"
    with open(tokenizer_path, 'w') as f:
        f.write(tokenizer.to_json())
    mlflow.log_artifact(tokenizer_path)

print("Model training and logging completed.")
