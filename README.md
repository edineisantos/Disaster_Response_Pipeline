# Disaster Response Pipeline

This project is part of the Data Scientist Nanodegree Program from Udacity. It involves building a machine learning pipeline to categorize real messages sent during disaster events. The goal is to help disaster response organizations classify messages so that they can direct resources where they are needed most.

## Table of Contents
- [Project Overview](#project-overview)
- [File Structure](#file-structure)
- [Instructions](#instructions)
- [Running the Scripts](#running-the-scripts)
- [Running the Web App](#running-the-web-app)
- [Conda Environment](#conda-environment)
- [Acknowledgements](#acknowledgements)

## Project Overview
In this project, we will process disaster response data and build a model that classifies disaster messages into categories. The steps include:
1. Data ETL Pipeline: Extract, Transform, and Load the data.
2. Machine Learning Pipeline: Build a model to classify messages.
3. Web App: Display the results and provide an interface to classify new messages.

## File Structure
```
README.md
├── app
│   ├── run.py
│   └── templates
│       ├── go.html
│       └── master.html
├── data
│   ├── DisasterResponseData.db
│   ├── disaster_categories.csv
│   ├── disaster_messages.csv
│   └── process_data.py
├── environment.yml
├── models
│   ├── classifier.pkl
│   └── train_classifier.py
└── notebooks
    ├── DisasterResponse.db
    ├── DisasterResponseData.db
    ├── ETL Pipeline Preparation.ipynb
    ├── ML Pipeline Preparation.ipynb
    ├── Twitter-sentiment-self-drive-DFE.csv
    ├── categories.csv
    └── messages.csv
```

## Instructions

### Running the Scripts
1. **ETL Pipeline**:
   - The script `process_data.py` is used for the ETL pipeline.
   - Command to run:
     ```sh
     python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponseData.db
     ```

2. **Machine Learning Pipeline**:
   - The script `train_classifier.py` is used for training the machine learning model.
   - Command to run:
     ```sh
     python models/train_classifier.py data/DisasterResponseData.db models/classifier.pkl
     ```

### Running the Web App
1. Navigate to the `app` directory:
   ```sh
   cd app
   ```

2. Run the Flask app:
   ```sh
   python run.py
   ```

3. Open your browser and go to:
   ```
   http://0.0.0.0:3000/
   ```

### Conda Environment
To ensure that you have all the necessary dependencies, you can create a conda environment using the provided `environment.yml` file.

1. Create the environment:
   ```sh
   conda env create -f environment.yml
   ```

2. Activate the environment:
   ```sh
   conda activate disaster_response
   ```

## Acknowledgements
This project was completed as part of the Udacity Data Scientist Nanodegree Program. The dataset was provided by [Figure Eight](https://appen.com/), and the project instructions were provided by Udacity.
