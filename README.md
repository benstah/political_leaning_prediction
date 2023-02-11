# Automatic Detection of the Political Leaning of Newspaper Articles


## Prerequisites:

This project uses **Python3**. Make sure it is installed before proceeding.

To make sure that all dependencies are installed run following command:
```pip3 install -r requirements.txt```

Run the ```src/setup/initialize.py``` script, to make sure that all the necessary packages are being downloaded.


## Training set
Includes weak labeled data
Leave out article for cross validation and fine tuninig 

  - Label **UNDEFINED** data
  - Leave out good amount and fine tune parameters step by step


## Validation set
Self annotated data (approx. 200)

  - Fine tune hyper paramters


## Test set
Self annotated data (approx. 200)

  - Test performance



### Directory Structure


    .
    |–– data                # Contains of all the datasets needed for the project
        |–– external        # Contains the original csv files
        |–– interim         # Unprocessed training, validation and test set
        |–– processed       # Processed datasets for modeling
        |–– raw             # Original data transferred to np files which are already including own annotations
    |–– data_preperation    # inlcudes scripts to prepare data from 'external' to 'raw'
    |–– src                 # Python project for modeling
        |–– models
        |–– preprocessing   # Preprocess interim data and make it processed data for training
        |–– utils           # Components that can be reused through out the project
        ...
    |–– requirements.txt    # Includes all dependencies that need to be installed