import sys
import pickle
# Data related libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
# NLP related libraries
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download(['punkt', 'stopwords', 'wordnet'])
# Modling related libraries
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    """
    load data from database and returns inputs, targets and targets names.
    Parameters:
    - database_filepath:str / path the database file.
    Return: 
    X:pandas.Series / Inputs.
    Y:pandas.Series / Targets.
    targets_names:list[str] / targets names(columns names).
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('CleanDataset', engine)
    X = df['message']
    Y = df.iloc[:, 6:]
    targets_names = Y.columns
    return X, Y, targets_names


def tokenize(text):
    """
    cleaning the text by changing the text to lower case, removing punctuation marks,
    tokenize the text, removing stopping words and stemming the text.
    Parameters:
    - text:str / text to be cleaned.
    Return: text:str / cleaned text.
    """
    # text to lower case
    text = text.lower()
    # removing punctuation marks
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    # tokenize the text
    text = word_tokenize(text)
    # removing stopping words
    text = [word for word in text if word not in stopwords.words('english')]
    # stemm the text
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text]
    
    return text


def build_model():
    """
    builds the machine learning model.
    Parameters: None
    Return: Machine learning model
    """
    pipeline = Pipeline([
        ('vectorize', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('classifier', OneVsRestClassifier(SVC()))
    ])
    parameters = {
        'classifier__estimator__C': (0.1, 1, 10)
    }

    cv = 2
    model = GridSearchCV(pipeline, param_grid=parameters, cv=cv, verbose=3)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the machine learning model and print its classification_report and Accuracy.
    Parameters:
    - model / machine learning model
    - X_test:np.array / testing set inputs
    - Y_test:np.array / testing set targets
    - category_names:list[str] / targets names(columns names)
    Return: None
    """
    y_pred = model.predict(X_test)
    report = classification_report(Y_test.values, y_pred, target_names=category_names, zero_division=1)
    print(report)
    print(f"Accuracy: {(np.mean(y_pred == Y_test.values) * 100)}%")


def save_model(model, model_filepath):
    """
    Saves the machine learning model into the given path.
    Parameters:
    - model / machine learning model.
    - model_filepath:str / path where the model will be saved. 
    Return: None
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


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