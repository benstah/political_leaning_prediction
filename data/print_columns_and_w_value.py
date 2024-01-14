from joblib import load, dump
import pandas as pd
import os

dirname = os.path.dirname(__file__)

# shows stats of all data sets
df_training_s = load(dirname + '/processed/training_set_s')

print("----------------- Training Set Small -----------------")
print("Rows: " + str(len(df_training_s.index)))
print(df_training_s.head())

print("\n----------values-----------\n")
train_df = df_training_s.loc[df_training_s["w2"]!=0.0]

print(train_df)
print(str(len(train_df.index)))