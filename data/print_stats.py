from joblib import load, dump
import pandas as pd
import os

dirname = os.path.dirname(__file__)

# shows stats of all data sets
df_training_s = load(dirname + '/processed/training_set_s')

print("----------------- Training Set Small -----------------")
print("Rows: " + str(len(df_training_s.index)))
print(df_training_s.head())
 
df_training_l = load(dirname + '/processed/training_set_l')

print("----------------- Training Set Large -----------------")
print("Rows: " + str(len(df_training_l.index)))
print(df_training_l.head())


df_gold_validation = load(dirname + '/processed/validation_set')

print("----------------- Gold Validation Set -----------------")
print("Rows: " + str(len(df_gold_validation.index)))
print(df_gold_validation.head())


df_test = load(dirname + '/processed/test_set')

print("----------------- Test Set -----------------")
print("Rows: " + str(len(df_test.index)))
print(df_test.head())