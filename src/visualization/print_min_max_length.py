from joblib import load
import matplotlib.pyplot as plt

import os

dirname = os.path.dirname(__file__)

def prepareLabels(df, label):
    headlines = df.headline.values.tolist()
    leads = df.lead.values.tolist()
    bodies = df.body.values.tolist()

    concats = [' '.join(item) for item in zip(headlines, leads, bodies)]

    len_min = float('inf')
    words_min = float('inf')
    len_max = 0
    words_max = 0

    for text in concats:
        length = len(text)

        if length > len_max:
            len_max = length
            words_max = text.count(' ')

        if length < len_min:
            len_min = length
            words_min = text.count(' ')

        

    print('\n')
    print(label)
    print('--------------------------------------------------------------------')
    print ("{:<15} {:<15} {:<15}".format(' ', 'Shortest','Longest'))
    print ("{:<15} {:<15} {:<15}".format('Characters', str(len_min),str(len_max)))
    print ("{:<15} {:<15} {:<15}".format('Words', str(words_min),str(words_max)))
    print ('\n')
           


title = 'Length of Articles (Head, Lead, Body) - Training Set S'
val_df = load(dirname + '/../../data/processed/training_set_s')
prepareLabels(val_df, 'Training Set S')


title = 'Length of Articles (Head, Lead, Body) - Validation Set L'
val_df = load(dirname + '/../../data/processed/training_set_l')
prepareLabels(val_df, 'Training Set L')


title = 'Length of Articles (Head, Lead, Body) - Validation Set'
val_df = load(dirname + '/../../data/processed/validation_set')
prepareLabels(val_df, 'Validation Set')


title = 'Length of Articles (Head, Lead, Body) - Test Set'
val_df = load(dirname + '/../../data/processed/test_set')
prepareLabels(val_df, 'Test Set')




