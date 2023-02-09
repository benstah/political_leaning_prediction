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
validationSet = []
goldValidationSet = []
testSet = []

# Only every second self rated article should be collected by [goldValidationSet] the other should be 
# collected by [testSet]. For identification an bool is being used, which will be changed every assignment
isTestSet = False

def handle_data_selection(data):
    global trainingSet
    global validationSet
    global goldValidationSet
    global testSet
    global isTestSet

    # loop through file
    for index, row in data.iterrows():
        if row.rating != 'UNDEFINED' and isTestSet == False:
            goldValidationSet.append(row)
            isTestSet = True

        elif row.rating != 'UNDEFINED' and isTestSet == True:
            testSet.append(row)
            isTestSet = False

        else:
            trainingSet.append(row)


#  load and seelct data for different data sets
df = load(dirname + '/2017_1')
handle_data_selection(df)

df = load(dirname + '/2018_2')
handle_data_selection(df)

df = load(dirname + '/2019_2')
handle_data_selection(df)



#  safe data sets to data directory
df_training = pd.DataFrame(trainingSet)
print("---------- Training set small ----------")
print(df_training.head())
filename = os.path.join(dirname, '../data/training_set_s')
dump(df_training, filename, compress=4)
# '../data/training_set', compress=4)

df_gold_validation = pd.DataFrame(goldValidationSet)
print("---------- Gold Validation set ----------")
print(df_gold_validation.head())
filename = os.path.join(dirname, '../data/validation_set')
dump(df_gold_validation, filename, compress=4)

df_test = pd.DataFrame(testSet)
print("---------- Test set ----------")
print(df_test.head())
filename = os.path.join(dirname, '../data/test_set')
dump(df_test, filename, compress=4)


