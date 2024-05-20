import sys
from sqlalchemy import create_engine
import pandas as pd
from sklearn.model_selection import train_test_split
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

def build_model():
    from sklearn.feature_extraction.text import HashingVectorizer
    # set the pipeline using HashingVectorizer
    model = Pipeline([
        ('vect', TfidfVectorizer(tokenizer=tokenize, max_df=0.75)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=30, min_samples_split=2)))
    ])

    return model

# setup dataframes for predicting 0 and 1 of the message
def f1_report_frame(category_names, y_test, y_pred, score_name='f1-score'):
    if score_name in ['f1-score', 'recall', 'precision']:
        f1_report = {}
        for i, col in enumerate(category_names):
            # Generate classification report as dictionary
            tmp_rpt = classification_report(y_test.iloc[:,i], y_pred[:,i], output_dict=True)
            # Extract F1-scores for each label
            f1_scores = {label: metrics[score_name] for label, metrics in tmp_rpt.items() if label not in ['accuracy', 'macro avg', 'weighted avg']}
            f1_report[col] = f1_scores
        print('The ' + score_name + ' is:')
        print(pd.DataFrame(f1_report))
    else:
        print("The score name need to be one of the three - 'f1-score','recall', or 'precision'")

def evaluate_model(model, X_test, y_test, category_names):
    # Evaluate the model
    y_pred = model.predict(X_test)
    # check the f1 score report
    f1_report_frame(category_names, y_test, y_pred)
    f1_report_frame(category_names, y_test, y_pred, score_name='recall')
    f1_report_frame(category_names, y_test, y_pred, score_name='precision')

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
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
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