# Automatic Detection of the Political Leaning of Newspaper Articles


## Prerequisites:

This project uses **Python3**. Make sure it is installed before proceeding.

To make sure that all dependencies are installed run following command:
```pip3 install -r requirements.txt```

Run the ```src/setup/initialize.py``` script, to make sure that all the necessary packages are being downloaded.

It is recommended to use a virtual environment to run this project.


## Models:
If you do not want to run the training yourself, you can reuse the models for the final test run.
They need to be added in the root directory of this project:

- best_model_base.pt (DistilBERT Baseline Model): https://ucloud.univie.ac.at/index.php/s/S3o2s7NYXNKz9La
- best_model_w1.pt (Many K-Fold Model): https://ucloud.univie.ac.at/index.php/s/ABQgy8rEbdKkawd
- best_model_w1_exposed.pt (Many Expressive K-Fold Model): https://ucloud.univie.ac.at/index.php/s/fRzT5i3aoeffygN
- best_model_w1_extreme_exposed.pt (Many Extremly Expressive K-Fold Model): https://ucloud.univie.ac.at/index.php/s/agCwr467bm9jrBw
- best_model_wa.pt (Few K-Fold Model): https://ucloud.univie.ac.at/index.php/s/dBoC7cTq4P27FyS
- best_model_wa_exposed.pt (Few Expressive K-Fold Model): https://ucloud.univie.ac.at/index.php/s/STCZNeT5EqaHqMG
- best_model_wa_extremly_exposed.pt (Few Extremly Expressive K-Fold Model): https://ucloud.univie.ac.at/index.php/s/AAXrYY5JPLLFyFn
- best_model_w20.pt (Leave-One-Out Model): https://ucloud.univie.ac.at/index.php/s/7KfqTAsnJ5PWrkK


## Training set
  - Includes weak labeled data
  - Label **UNDEFINED** data is filtered out for training

## Validation set
Self annotated data (approx. 200)


## Test set
Self annotated data (approx. 200)



### Directory Structure


    .
    |–– data                 # Contains of all the datasets needed for the project
        |–– external         # Contains the original csv files
        |–– interim          # Unprocessed training, validation and test set
        |–– processed        # Processed datasets for logistic regression baseline
        |–– processed_vector # Processed datasets for modeling
        |–– raw              # Original data transferred to np files which are already including own annotations
        |–– README.md        # Contains all the download and view links to the datasets
    |–– data_preperation     # inlcudes scripts to prepare data from 'external' to 'raw'
    |–– src                  # Python project for modeling
        |–– models           # Contains all the prediction and training models
        |–– trainer          # Contains all the training scripts
        |–– preprocessing    # Preprocess interim data and make it processed data for training
        |–– utils            # Components that can be reused through out the project
        |–– visualization    # Contains all the plot scripts
        ...
    |–– requirements.txt     # Includes all dependencies that need to be installed