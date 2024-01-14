import pandas as pd
from joblib import load
import os
import csv

dirname = os.path.dirname(__file__)

# # add for test_set
filename = dirname + '/../data/processed/test_set'
df = load(filename)
# df['label_id'] = df["rating"].map(lambda x: _numerize_labels(x))
# print(df.head)

with open(dirname + "/../baseline_submission.csv", "w", newline='') as file:
    writer = csv.writer(file)

    writer.writerow(['id', 'political_leaning', 'rating', 'label_id' , 'target'])

    for index, row in df.iterrows():
        writer.writerow([row['id'], row['political_leaning'], row['rating'], row['label_id'], 9])

