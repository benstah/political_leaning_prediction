from joblib import load, dump
import pandas as pd
import os

dirname = os.path.dirname(__file__)

# shows stats of all data sets
df_training_s = load(dirname + '/training_set_s.pkl')

print("----------------- Training Set Small -----------------")
print("Rows: " + str(len(df_training_s.index)))
print(df_training_s.head())
 
df_training_l = load(dirname + '/training_set_l.pkl')

print("----------------- Training Set Large -----------------")
print("Rows: " + str(len(df_training_l.index)))
print(df_training_l.head())


df_gold_validation = load(dirname + '/validation_set.pkl')

print("----------------- Gold Validation Set -----------------")
print("Rows: " + str(len(df_gold_validation.index)))
print(df_gold_validation.head())


df_test = load(dirname + '/test_set.pkl')

print("----------------- Test Set -----------------")
print("Rows: " + str(len(df_test.index)))
print(df_test.head())