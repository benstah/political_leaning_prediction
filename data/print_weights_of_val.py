from joblib import load, dump
import pandas as pd
import os

dirname = os.path.dirname(__file__)


df = load(dirname + '/processed/training_set_s')

print(str(len(df.loc[df["w1"] == 0])))
print(str(len(df.loc[df["w2"] == 0])))
print(str(len(df.loc[df["w3"] == 0])))
print(df["label_id"].value_counts())

print(df.head())