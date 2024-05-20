# Disaster Response Pipeline

## Project Overview
The goal is to build a machine learning pipeline to categorize disaster messages. These messages are sent during disasters to aid organizations and must be classified quickly to ensure that appropriate aid is dispatched.

The project involves:
1. Data ETL pipeline: Extract, Transform, Load
2. Machine Learning pipeline to classify messages
3. Web application to showcase the model

## Installation
To run the project, ensure you have the following packages installed:

- Python 3.11
- pandas
- numpy
- sqlalchemy
- scikit-learn
- nltk
- flask
- plotly

## File Descriptions
* data/process_data.py: Script to process the data.
* models/train_classifier.py: Script to train the machine learning model.
* app/run.py: Script to run the web application.
* data/disaster_messages.csv: Dataset containing messages.
* data/disaster_categories.csv: Dataset containing message categories.

## Machine Learning Pipeline
The machine learning pipeline (train_classifier.py) includes:

- Text processing and feature extraction using TfidfVectorizer
- Multi-output classification using RandomForestClassifier
- Hyperparameter tuning using GridSearchCV
- Web Application
- The Flask web application (run.py) allows users to input a message and get classification results in real-time. It also displays visualizations of the training data.

## Instructions
### Run ETL Pipeline: This step processes the data.
> python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/sql_database.db

### Run ML Pipeline: This step trains the model.
> python models/train_classifier.py data/sql_database.db models/disaster_response_model.pkl

### Run the Web App: This step starts the web application.
> python app/run.py
> Then go to http://localhost:3001/ (e.g., http://127.0.0.1:3000) in your browser.
