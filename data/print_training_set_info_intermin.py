from joblib import load, dump
import pandas as pd
import os

dirname = os.path.dirname(__file__)


df_training = load(dirname + '/interim/training_set_s')

print("----------------- Gold Validation Set -----------------")
print("Rows: " + str(len(df_training.index)))
print(df_training.head())
print(df_training["rating"].value_counts())
print(df_training.lead)
# print(df_training["label_id"].value_counts())
print(df_training.columns.tolist())

