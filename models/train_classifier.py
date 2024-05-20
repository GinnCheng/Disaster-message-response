"""
train_classifiers.py

This script trains a machine learning model to classify disaster messages into multiple categories.
The model is trained using a pipeline that includes text preprocessing (tokenization, lemmatization),
feature extraction (TF-IDF), and a multi-output classifier (RandomForestClassifier).

The script performs the following steps:
1. Load and clean the dataset.
2. Define a machine learning pipeline.
3. Train the model using GridSearchCV to find the best parameters.
4. Evaluate the model on a test set.
5. Save the trained model to a file for future use.

Functions:
-----------
- load_data(database_filepath): Load the dataset from the SQLite database.
- tokenize(text): Preprocess text by normalizing, tokenizing, and lemmatizing.
- build_model(): Build a machine learning pipeline and perform a grid search.
- evaluate_model(model, X_test, Y_test, category_names): Evaluate the model and print the classification report.
- save_model(model, model_filepath): Save the trained model to a pickle file.
- main(): Main function to run the script.

Parameters:
-----------
- database_filepath (str): Filepath of the SQLite database containing the cleaned dataset.
- model_filepath (str): Filepath where the trained model will be saved.

Usage:
------
To run the script from the command line:
$ python train_classifiers.py <database_filepath> <model_filepath>

Example:
--------
$ python models/train_classifier.py data/sql_database.db models/disaster_response_model.pkl
"""


import sys
from sqlalchemy import create_engine
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from nltk.stem import WordNetLemmatizer
import re
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('sql_database', con=engine)
    # train and test split
    X = df.message
    y = df.loc[:, 'related':'direct_report']
    return X, y, y.columns

def tokenize(text):
    # Remove punctuation and non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove words with 1 or 2 characters
    tokenized_text = word_tokenize(text)
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokenized_text = [lemmatizer.lemmatize(token) for token in tokenized_text]
    return tokenized_text

def build_model(X_train, Y_train):
    # setup the pipeline
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    # setup the grid search
    parameters = {
        'clf__estimator__n_estimators': [5, 10],
        'clf__estimator__min_samples_split': [2, 4]
    }
    cv = 3
    grid_search = GridSearchCV(pipeline, param_grid=parameters, cv=cv)
    print('Training model...')
    grid_search.fit(X_train, Y_train)
    # Print best parameters
    print("Best parameters found:")
    print(grid_search.best_params_)
    # Use the best estimator to make predictions
    best_pipeline = grid_search.best_estimator_

    return best_pipeline

# setup dataframes for predicting 0 and 1 of the message
def report_frame(category_names, y_test, y_pred):
    for i, col in enumerate(category_names):
        print(f"Category: {col}\n", classification_report(y_test[col], y_pred[:, i]))

def evaluate_model(model, X_test, y_test, category_names):
    # Evaluate the model
    y_pred = model.predict(X_test)
    # check the f1 score report
    report_frame(category_names, y_test, y_pred)

def save_model(model, model_filepath):
    import joblib
    # Save the model
    joblib.dump(model, model_filepath)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X_train, Y_train)
        
        # print('Training model...')
        # model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()