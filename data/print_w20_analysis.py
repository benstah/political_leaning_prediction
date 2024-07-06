from joblib import load, dump
import pandas as pd
import os

dirname = os.path.dirname(__file__)

# shows stats of all data sets
df_training_s = load(dirname + '/processed/training_set_s')

print("----------------- Training Set Small -----------------")
df_training_s = df_training_s.loc[df_training_s["political_leaning"]!="UNDEFINED"]
df_training_s = df_training_s[df_training_s["w20"].notnull()]

print(df_training_s["outlet"].unique())

df_training_s = df_training_s[["outlet", "label_id", "political_leaning", "w20"]].drop_duplicates()

print(df_training_s.head())

