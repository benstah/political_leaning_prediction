from joblib import load, dump
import pandas as pd
import os

dirname = os.path.dirname(__file__)
df = load(dirname + '/../data/processed/training_set_s')

df['w1'] = 0
df['w2'] = 0
df['w3'] = 0
df['wa'] = 0
print(df.head())

dump(df, dirname + '/../data/processed/training_set_s', compress=4)
print(df.head())