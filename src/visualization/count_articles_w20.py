from joblib import load
import matplotlib.pyplot as plt

import pandas as pd
import os

dirname = os.path.dirname(__file__)

def prepareLabels(df):
    outlets = df.outlet.values.tolist()

    article_count = {}

    all_outlets = []

    for outlet in outlets:
        article_count[outlet] = article_count.get(outlet, 0) + 1

    counts = []
    bar_colors = []

    for key in article_count:
        counts.append(article_count[key])

        leaning = df[df.outlet == key].iloc[0].political_leaning
        outlet = key + '\n' + leaning
        all_outlets.append(outlet)

        if leaning == 'CENTER':
            bar_colors.append('tab:green')
        elif leaning == 'LEFT':
            bar_colors.append('tab:blue')
        elif leaning == 'RIGHT':
            bar_colors.append('tab:red')
        
    return counts, bar_colors, all_outlets


def plotStats(counts, bar_colors, title, outlets):
    fig, ax = plt.subplots()
    articles = outlets
    bar_container = ax.bar(articles, counts, color=bar_colors)
    ax.bar_label(bar_container, fmt='{:,.0f}')

    ax.set_ylabel('Count of Articles')
    ax.set_title(title)

    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='center')

    plt.show()


title = 'Count of Articles per Outlet - Training Set S'
val_df = load(dirname + '/../../data/processed/training_set_s')
val_df = val_df.loc[val_df["political_leaning"]!="UNDEFINED"]
val_df = val_df[val_df["w20"].notnull()]
# outlets = ['NBC News', 'ABC News', 'Fox News', 'NPR', 'Los Angeles Times', 'The Guardian', 'USA Today','Breitbart','HuffPost', 'Slate', 'CBS News', 'The New York Times', 'Reuters']
counts, bar_colors, outlets = prepareLabels(val_df)
plotStats(counts, bar_colors, title, outlets)

