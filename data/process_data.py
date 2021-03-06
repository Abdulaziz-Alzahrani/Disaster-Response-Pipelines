import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Reads the csv data from two files and load them into a single dataframe.
    Parameters:
    - messages_filepath:str / the path to messages.csv file.
    - categories_filepath:str / the path to categories.csv file.
    Return: pandas.Dataframe
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = pd.merge(messages, categories, on='id')
    
    return df


def clean_data(df):
    """
    Cleans the dataframe by adding appropriate columns, renaming them and dropping duplicates
    if they exist.
    Parameters:
    - df:pandas.Dataframe / the dataframe of merged datasets that needs to be cleaned.
    Return: pandas.Dataframe / Cleaned DataFrame.
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    categories.head()
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # extract a list of new column names for categories.
    # this is done using lambda function that takes everything 
    # up to the second to last character of each string with slicing
    get_name_func = lambda x: x[:-2]
    category_colnames = [get_name_func(x) for x in row]
    print(category_colnames)
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    
    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # since 'related' column has (0,1,2), 2 will be considerd as 1 to make it binary
    df['related'].replace(2, 1, inplace=True)

    if df.duplicated().sum()>0: # check number of duplicates
        # drop duplicates
        df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):
    """
    Saves the dataframe into a SQLite database, given its file name.
    Parameters:
    - df:pandas.Dataframe / the dataframe of cleaned dataset.
    - database_filename:str / name of the database file
    Return: None
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('CleanDataset', engine, index=False, if_exists='replace')  


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