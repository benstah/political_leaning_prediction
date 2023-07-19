from joblib import load, dump
import pandas as pd
import os

# read file per file into scirpt
# collect data with weakly labeled data and no self rated label and create file (training set)
# collect data with undefined label and no self rated label and create file (validation set)
# collect every second self rated article and create file (gold validation set)
# collect every other second self rated article and create file (test set)



dirname = os.path.dirname(__file__)


def _numerize_labels (political_leaning):
        # RIGHT = 0
        # LEFT = 1
        # CENTER = 2
        # UNDEFINED = 3
        # Nothing from the above = 4

        # for leaning in political_leaning:
        if political_leaning == "RIGHT":
            return 0
        elif political_leaning == "LEFT":
            return 1
        elif political_leaning == "CENTER":
            return 2
        elif political_leaning == "UNDEFINED":
            return 3
        else:
            return 4


# add for training_set_s
filename = dirname + '/../data/processed/training_set_s'
df = load(filename)
df['label_id'] = df["political_leaning"].map(lambda x: _numerize_labels(x))
print(df.head)
dump(df, filename, compress=4)

# add for training_set_l
# filename = dirname + '/../data/processed/training_set_l'
# df = load(filename)
# df['label_id'] = df["political_leaning"].map(lambda x: _numerize_labels(x))
# print(df.head)
# dump(df, filename, compress=4)

# # add for validation_set
# filename = dirname + '/../data/processed/validation_set'
# df = load(filename)
# df['label_id'] = df["rating"].map(lambda x: _numerize_labels(x))
# print(df.head)
# dump(df, filename, compress=4)

# # add for test_set
# filename = dirname + '/../data/processed/test_set'
# df = load(filename)
# df['label_id'] = df["rating"].map(lambda x: _numerize_labels(x))
# print(df.head)
# dump(df, filename, compress=4)




