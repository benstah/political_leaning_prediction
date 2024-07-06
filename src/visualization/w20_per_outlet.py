from joblib import load
import matplotlib.pyplot as plt

import os

dirname = os.path.dirname(__file__)

import pandas as pd

def prepareLabels(df):
    # Drop duplicate rows based on 'outlet' column and keep the first occurrence
    df_unique_outlets = df.drop_duplicates(subset='outlet')

    # Extract outlet names and their corresponding 'w20' values
    outlets = df_unique_outlets['outlet'].tolist()
    percentages = df_unique_outlets['w20'].tolist()
    political_leaning = df_unique_outlets['political_leaning'].tolist()

    all_outlets = ['\n'.join(item) for item in zip(outlets, political_leaning)]

    # Setting uniform color for all bars
    bar_colors = []
    # bar_colors = ['tab:blue'] * len(all_outlets)
    for outlet in all_outlets:
        if 'CENTER' in outlet:
            bar_colors.append('tab:green')
        elif 'LEFT' in outlet:
            bar_colors.append('tab:blue')
        elif 'RIGHT' in outlet:
            bar_colors.append('tab:red')
    
    return percentages, bar_colors, all_outlets



def plotStats(percentages, bar_colors, title, outlets):
    fig, ax = plt.subplots()
    articles = outlets
    bar_container = ax.bar(articles, percentages, color=bar_colors)
    ax.bar_label(bar_container, fmt='{:,.2f}')

    ax.set_ylabel('Percentage in (%)')
    ax.set_title(title)

    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='center')

    plt.show()


title = 'Weight of Sample per Outlet - Training Set S'
val_df = load(dirname + '/../../data/processed/training_set_s')
val_df = val_df.loc[val_df["political_leaning"]!="UNDEFINED"]
val_df = val_df[val_df["w20"].notnull()]
# outlets = ['NBC News', 'ABC News', 'Fox News', 'NPR', 'Los Angeles Times', 'The Guardian', 'USA Today','Breitbart','HuffPost', 'Slate', 'CBS News', 'The New York Times', 'Reuters']
percentages, bar_colors, outlets = prepareLabels(val_df)
plotStats(percentages, bar_colors, title, outlets)

