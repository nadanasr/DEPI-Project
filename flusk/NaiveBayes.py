import re
import nltk
import mlflow
import mlflow.sklearn
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

# Ensure NLTK resources are downloaded
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

# MLflow experiment setup
mlflow.set_experiment("Customer_service_intent_classification")

def preprocess_text(text):
    """Clean and preprocess text data."""
    text = text.lower()  # Lowercase text
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

def main():
    # Load data
    Train_PATH = os.path.join(os.getcwd(), 'Training_Dataset.csv')
    df_train = pd.read_csv(Train_PATH)

    Val_PATH = os.path.join(os.getcwd(), 'Validation_Dataset.csv')
    df_val = pd.read_csv(Val_PATH)

    Test_PATH = os.path.join(os.getcwd(), 'Training_Dataset.csv')
    df_test = pd.read_csv(Test_PATH)


    # Preprocess the data
    df_train['utterance'] = df_train['utterance'].apply(preprocess_text)
    df_test['utterance'] = df_test['utterance'].apply(preprocess_text)
    df_val['utterance'] = df_val['utterance'].apply(preprocess_text)

    # Prepare training data
    train_texts = df_train['utterance'].tolist()
    train_labels = df_train['intent'].tolist()
    # test_texts = df_test['utterance'].values
    # test_labels = df_test['intent'].values
    # X_val = df_val["utterance"].to_list()
    # y_val = df_val["intent"].to_list()

    # Tokenize the text
    train_texts = [text.split() for text in train_texts]
    train_texts = [" ".join(text) for text in train_texts]

    # Vectorize the text
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(train_texts)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, train_labels, test_size=0.2, random_state=42)

    # Start MLflow run
    with mlflow.start_run(run_name = "NaiveBayes"):
        # Train the Naive Bayes classifier
        classifier = MultinomialNB()
        classifier.fit(X_train, y_train)

        # Predict on the test set
        y_pred = classifier.predict(X_test)

        # Calculate accuracy and f1-score
        f1 = f1_score(y_test, y_pred, average='macro')
        accuracy = accuracy_score(y_test, y_pred)

        # Log the metrics and parameters to MLflow
        mlflow.log_param("model_type", "NaiveBayes")
        mlflow.log_param("vectorizer", "CountVectorizer")
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", accuracy)

        # Log the model
        mlflow.sklearn.log_model(classifier, "naive_bayes_model")

        print(f"Model trained with F1 score: {f1:.2f} and accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()

    