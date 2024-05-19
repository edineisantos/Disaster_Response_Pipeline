"""
This script processes disaster response data.

It saves the cleaned data into an SQLite database.

The script performs the following steps:
1. Loads the messages and categories datasets.
2. Merges the datasets.
3. Cleans the data by splitting categories into separate columns, converting 
   values to binary, and removing duplicates.
4. Saves the cleaned data into an SQLite database.

Usage:
    python process_data.py disaster_messages.csv disaster_categories.csv 
    DisasterResponse.db
"""

import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories datasets and merge them.

    Args:
    messages_filepath (str): File path to the messages dataset.
    categories_filepath (str): File path to the categories dataset.

    Returns:
    df (DataFrame): Merged DataFrame of messages and categories.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    """
    Clean the merged DataFrame by splitting categories into separate columns.
    
    Converts values to binary and removes duplicates.

    Args:
    df (DataFrame): Merged DataFrame of messages and categories.

    Returns:
    df (DataFrame): Cleaned DataFrame.
    """
    # Create a dataframe of the 36 individual category columns
    categories_expanded = df['categories'].str.split(';', expand=True)

    # Select the first row of the categories dataframe
    row = categories_expanded.iloc[0]

    # Extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x.split('-')[0])

    # Rename the columns of `categories`
    categories_expanded.columns = category_colnames

    for column in categories_expanded:
        # Set each value to be the last character of the string
        categories_expanded[column] = categories_expanded[column].str[-1:]

        # Convert column from string to numeric
        categories_expanded[column] = categories_expanded[column].astype(int)

    # Drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories_expanded], axis=1)

    # Drop rows where any category column has a value other than 0 or 1
    for column in category_colnames:
        df = df[df[column].isin([0, 1])]

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filepath):
    """
    Save the clean dataset into an SQLite database.

    Args:
    df (DataFrame): Cleaned DataFrame.
    database_filepath (str): File path for the SQLite database.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql('Message', engine, index=False, if_exists='replace')


def main():
    """
    Run the data processing functions: load, clean, and save data.

    This function executes the ETL pipeline:
    1. Loads the data from the provided file paths.
    2. Cleans the data.
    3. Saves the cleaned data to an SQLite database.
    """
    if len(sys.argv) == 4:

        (
            messages_filepath,
            categories_filepath,
            database_filepath
        ) = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
