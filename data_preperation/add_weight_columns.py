from joblib import load, dump
import pandas as pd
import os

dirname = os.path.dirname(__file__)
df = load(dirname + '/../data/processed/training_set_s')

# df['w1'] = 0
df['w2'] = 0
df['w3'] = 0
df['w4'] = 0
df['w5'] = 0
df['w6'] = 0
df['w7'] = 0
df['w8'] = 0
df['w9'] = 0
df['w10'] = 0
df['wa'] = 0
print(df.head())

dump(df, dirname + '/../data/processed/training_set_s', compress=4)
print(df.head())


# df = load(dirname + '/../data/processed/training_set_l')

# df['w1'] = 0
# df['w2'] = 0
# df['w3'] = 0
# df['wa'] = 0
# print(df.head())

# dump(df, dirname + '/../data/processed/training_set_l', compress=4)
# print(df.head())


# df = load(dirname + '/../data/processed/validation_set')

# df['w1'] = 0
# df['w2'] = 0
# df['w3'] = 0
# df['wa'] = 0
# print(df.head())

# dump(df, dirname + '/../data/processed/validation_set', compress=4)
# print(df.head())


# df = load(dirname + '/../data/processed/test_set')

# df['w1'] = 0
# df['w2'] = 0
# df['w3'] = 0
# df['wa'] = 0
# print(df.head())

# dump(df, dirname + '/../data/processed/test_set', compress=4)
# print(df.head())