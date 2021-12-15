# Disaster-Response-Pipelines

## Project Summary:
Disaster response is important where governments and companies take such matter seriously as it can save people,properties, etc...
So giving people, governments and companies a heads up when a disaster occures is very important, as the appropiate action to the 
specific type of disaster results in higher readiness, better warnings, and decreased susceptibility.

This project focuses on building a "Disaster Response Pipeline", which classifies disaster messages by filtering messages, tweets and any text
put on social media into categories using a supervised machine learning model trained on `Figure Eight` dataset.
This project is made from three components which are:
### A) ETL Pipeline:
In this section the data is extracted from two datasets where it gets merged, cleaned and stored in SQLite database.
### B) ML Pipeline:
In this section the dataset is loaded and splited to train/test dataset then text processing and a machine learning 
pipeline is built.
After that the model is trained and tuned using Gridsearch, then gets exported into a pickle file.
### C) Flask app:
In this section the model is deployed using a flask app where the dataset classes and thier count are displayed and 
the user can Enter a message where the model should classify the message.

## Project Content
### Disaster-Response-Pipelines
The project folder that contains the following
#### app folder:
This folder contains "run.py" where the flask app code is written and "templates" folder which contains the html pages
required by the app.
#### data folder:
This folder containes the two datasets "disaster_messages" and "disaster_categories" in csv format also "process_data.py"
which builds the ETL pipeline.
#### model folder:
This folder containes "train_classifier.py" which builds the text processing and machine learning Pipelines and export
the model in a pickle format.
#### Readme.md file:
A discreption of the project.
#### Licence:
MIT Licence.

## How to Experiment with the Project:
Before showing how to use the scripts lets go through the required libraies:
### 1- panadas
```bash
pip install pandas
```
### 2- numpy
```bash
pip install numpy
```
### 3- joblib
```bash
pip install joblib
```
### 4- nltk
```bash
pip install nltk
```
### 5- plotly
```bash
pip install plotly
```
### 6- regex
```bash
pip install regex
```
### 7- scikit-learn
```bash
pip install scikit-learn
```
### 6- SQLAlchemy
```bash
pip install sqlalchemy
```
 After the installation of the libraries go ahead and move to data directory via the commad line, then run the following 
 command.

```bash
python process_data.py disaster_messages.csv disaster_categories.csv Dataset.db
```
Note that you can use any other name for the database rather than "Dataset".
After the running "process_data.py" you can now move to model directory and run the following command.
```bash
python train_classifier.py ../data/Dataset.db Classifier.pickle
```
Note that you need to use the name you chose for the database and you can change the name of your model
prefer to use something else rather than "Classifier".

Now you can go to app directory and run the flask web app using run the following command.
```bash
python run.py
```
To open the project on your browser copy the link from your cmd next to:
 _* Running on http://0.0.0.0:3001/ (Press CTRL+C to quit)_
 
 and paste in your browser.
 
 ## Acknowledgements 

I would like to express my gratitude to Misk Academy and Udacity for this Amazing program
that expanded my knowledge and helped me in making this project.
