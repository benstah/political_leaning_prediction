from joblib import load
import matplotlib.pyplot as plt

import os

dirname = os.path.dirname(__file__)

def prepareLabels(df, label):
    leanings = df.political_leaning.values.tolist()
    labels = df.rating.values.tolist()

    leftToLeft = 0
    leftToRight = 0
    leftToCenter = 0

    rightToRight = 0
    rightToLeft = 0
    rightToCenter = 0

    centerToCenter = 0
    centerToRight = 0
    centerToLeft = 0

    undefinedToCenter = 0
    undefinedToRight = 0
    undefinedToLeft = 0

    for idx, val in enumerate(leanings):
        if val == 'RIGHT':
            if labels[idx] == 'RIGHT':
                rightToRight = rightToRight + 1
            elif labels[idx] == 'LEFT':
                rightToLeft = rightToLeft + 1
            else:
                rightToCenter = rightToCenter + 1
        elif val == 'LEFT':
            if labels[idx] == 'RIGHT':
                leftToRight = leftToRight + 1
            elif labels[idx] == 'LEFT':
                leftToLeft = leftToLeft + 1
            else:
                leftToCenter = leftToCenter + 1
        elif val == 'CENTER':
            if labels[idx] == 'RIGHT':
                centerToRight = centerToRight + 1
            elif labels[idx] == 'LEFT':
                centerToLeft = centerToLeft + 1
            else:
                centerToCenter = centerToCenter + 1
        elif val == 'UNDEFINED':
            if labels[idx] == 'RIGHT':
                undefinedToRight = undefinedToRight + 1
            elif labels[idx] == 'LEFT':
                undefinedToLeft = undefinedToLeft + 1
            else:
                undefinedToCenter = undefinedToCenter + 1

    print('\n')
    print(label)
    print('--------------------------------------------------------------------')
    print ("{:<15} {:<15} {:<15} {:<15}".format(' ', 'To Left','To Center', 'To Right'))
    print ("{:<15} {:<15} {:<15} {:<15}".format('Left', str(leftToLeft),str(leftToCenter), str(leftToRight)))
    print ("{:<15} {:<15} {:<15} {:<15}".format('Center', str(centerToLeft),str(centerToCenter), str(centerToRight)))
    print ("{:<15} {:<15} {:<15} {:<15}".format('Right', str(rightToLeft),str(rightToCenter), str(rightToRight)))
    print ("{:<15} {:<15} {:<15} {:<15}".format('Undefined', str(undefinedToLeft),str(undefinedToCenter), str(undefinedToRight)))
    print ('\n')
           


title = 'Length of Articles (Head, Lead, Body) - Validation Set'
val_df = load(dirname + '/../../data/processed/validation_set')
prepareLabels(val_df, 'Validation Set')


title = 'Length of Articles (Head, Lead, Body) - Test Set'
val_df = load(dirname + '/../../data/processed/test_set')
prepareLabels(val_df, 'Test Set')




