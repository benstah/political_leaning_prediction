from joblib import load, dump
import pandas as pd
import os

dirname = os.path.dirname(__file__)
df = load(dirname + '/../data/raw/2017_2')

df['rating'] = 'UNDEFINED'
print(df.head())

dump(df, dirname + '/../data/raw/2017_2', compress=4)
print(df.head())