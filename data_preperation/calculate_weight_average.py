from joblib import load, dump
import pandas as pd
import os

dirname = os.path.dirname(__file__)
df = load(dirname + '/../data/processed/training_set_s')

for index, row in df.iterrows():
    if row.wa == 0 and row.political_leaning != "UNDEFINED":
        df.at[index, 'wa'] = (row.w2 + row.w3 + row.w4 + row.w5 + row.w6 + row.w7 + row.w8 + row.w9) / 8

dump(df, dirname + '/../data/processed/training_set_s', compress=4)
print(df.head())
