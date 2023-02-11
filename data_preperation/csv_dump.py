import pandas as pd
from joblib import dump
import os

dirname = os.path.dirname(__file__)

df = pd.read_csv(dirname + '/../data/external/2017_2.csv')

print(df.head())

dump(df, dirname + '/../data/raw/2017_2', compress=4)