import pandas as pd
from joblib import dump
import os

dirname = os.path.dirname(__file__)

df = pd.read_csv(dirname + '/2017_2.csv')

print(df.head())

dump(df, dirname + '/2017_2', compress=4)