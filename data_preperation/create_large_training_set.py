from joblib import load, dump
import pandas as pd
import os

# read file per file into scirpt
# collect data with weakly labeled data and no self rated label and create file (training set)
# collect data with undefined label and no self rated label and create file (validation set)
# collect every second self rated article and create file (gold validation set)
# collect every other second self rated article and create file (test set)



dirname = os.path.dirname(__file__)

trainingSet = []


def handle_data_selection(data):
    global trainingSet

    # loop through file
    for index, row in data.iterrows():
        if row.rating == 'UNDEFINED':
            trainingSet.append(row)


#  load and seelct data for different data sets
df = load(dirname + '/../data/raw/2017_1')
handle_data_selection(df)

df = load(dirname + '/../data/raw/2017_2')
handle_data_selection(df)

df = load(dirname + '/../data/raw/2018_1')
handle_data_selection(df)

df = load(dirname + '/../data/raw/2018_2')
handle_data_selection(df)

df = load(dirname + '/../data/raw/2019_1')
handle_data_selection(df)

df = load(dirname + '/../data/raw/2019_2')
handle_data_selection(df)



#  safe data sets to data directory
df_training = pd.DataFrame(trainingSet)
print("---------- Training set large ----------")
print(df_training.head())
filename = os.path.join(dirname, '/../data/interim/training_set_l')
dump(df_training, filename, compress=4)


