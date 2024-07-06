from joblib import load, dump
import pandas as pd
import os

dirname = os.path.dirname(__file__)

# shows stats of all data sets
df_training_s = load(dirname + '/processed/training_set_s')

print("----------------- Training Set Small -----------------")
df_training_s = df_training_s.loc[df_training_s["political_leaning"]!="UNDEFINED"]
print(df_training_s.w1.min())
print(df_training_s.w1.max())
