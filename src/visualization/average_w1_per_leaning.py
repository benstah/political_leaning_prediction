from joblib import load
import matplotlib.pyplot as plt

import os

dirname = os.path.dirname(__file__)

import pandas as pd

def prepareLabels(df):
    # Grouping by 'outlet' and summing the 'w1' column to get article count for each outlet
    outlet_counts = df.groupby('political_leaning')['w1'].mean()

    # Extracting outlet names and their corresponding article counts
    all_outlets = outlet_counts.index.tolist()
    percentages = outlet_counts.tolist()

    # Setting uniform color for all bars
    bar_colors = ['tab:blue'] * len(all_outlets)
    
    return percentages, bar_colors, all_outlets




def plotStats(percentages, bar_colors, title, outlets):
    fig, ax = plt.subplots()
    articles = outlets
    bar_container = ax.bar(articles, percentages, color=bar_colors)
    ax.bar_label(bar_container, fmt='{:,.4f}')

    ax.set_ylabel('Percentage in (%)')
    ax.set_title(title)

    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='center')

    plt.show()


title = 'Weight of Sample per Outlet - Training Set S (Small folds)'
val_df = load(dirname + '/../../data/processed/training_set_s')
val_df = val_df.loc[val_df["political_leaning"]!="UNDEFINED"]
val_df = val_df[val_df["w20"].notnull()]
# outlets = ['NBC News', 'ABC News', 'Fox News', 'NPR', 'Los Angeles Times', 'The Guardian', 'USA Today','Breitbart','HuffPost', 'Slate', 'CBS News', 'The New York Times', 'Reuters']
percentages, bar_colors, outlets = prepareLabels(val_df)
plotStats(percentages, bar_colors, title, outlets)

