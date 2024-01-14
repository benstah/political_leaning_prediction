from joblib import load, dump
import pandas as pd
import os

dirname = os.path.dirname(__file__)


df_gold_validation = load(dirname + '/processed/validation_set')

print("----------------- Gold Validation Set -----------------")
print("Rows: " + str(len(df_gold_validation.index)))
print(df_gold_validation.head())
print(df_gold_validation["rating"].value_counts())
print(df_gold_validation["label_id"].value_counts())
print(df_gold_validation.columns.tolist())

df_test = load(dirname + '/processed/test_set')

print("----------------- Test Set -----------------")
print("Rows: " + str(len(df_test.index)))
print(df_test.head())
print(df_test["rating"].value_counts())
print(df_test["label_id"].value_counts())
