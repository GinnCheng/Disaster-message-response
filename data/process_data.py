"""
process_data.py

This script processes disaster response data and stores it in a SQLite database. The data
is loaded from CSV files, cleaned, and then saved into a database for later use in machine
learning pipelines.

The script performs the following steps:
1. Load messages and categories datasets.
2. Merge the datasets on the 'id' column.
3. Clean the merged dataset by splitting the categories and converting them into binary values.
4. Remove duplicates and save the cleaned data into a SQLite database.

Functions:
-----------
- load_data(messages_filepath, categories_filepath): Load and merge messages and categories datasets.
- clean_data(df): Clean the merged dataset.
- save_data(df, database_filepath): Save the cleaned data into a SQLite database.
- main(): Main function to run the script.

Parameters:
-----------
- messages_filepath (str): Filepath of the messages CSV file.
- categories_filepath (str): Filepath of the categories CSV file.
- database_filepath (str): Filepath where the SQLite database will be saved.

Usage:
------
To run the script from the command line:
$ python process_data.py <messages_filepath> <categories_filepath> <database_filepath>

Example:
--------
$ python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/sql_database.db
"""

import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
pd.set_option('display.max_columns',20)

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath, index_col='id')
    categories = pd.read_csv(categories_filepath, index_col='id')
    # split the categories values
    categories = categories.categories.str.split(pat=';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0, :]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x.split('-')[0]).values

    # rename the columns of `categories`
    categories.columns = category_colnames
    categories.index.name = 'id'

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split(pat='-', expand=True).iloc[:, -1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # make sure all values are 0 or 1
    categories[categories == 2] = 1
    df =messages.merge(categories, on='id')

    return df

def clean_data(df):
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    sql_loc = 'sqlite:///' + database_filename
    engine = create_engine(sql_loc)
    df.to_sql('sql_database', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()