from joblib import load
import matplotlib.pyplot as plt

import os

dirname = os.path.dirname(__file__)

def prepareLabels(df):
    outlets = df.outlet.values.tolist()
    count_total = len(outlets)

    article_count = {}
    article_percentage = {}

    all_outlets = []
    outlet_color = {}

    for outlet in outlets:
        article_count[outlet] = article_count.get(outlet, 0) + 1
        if outlet not in article_percentage:
            article_percentage[outlet] = 0.0    

        
    percentages = []
    counts = []
    bar_labels = []
    bar_colors = []

    for key in article_count:
        article_percentage[key] = article_count[key] / count_total * 100
        percentages.append(article_percentage[key])
        counts.append(article_count[key])
        all_outlets.append(key)
        
        
        first_outlet_entry = df.loc[df["outlet"]==key].iloc[0]
        if first_outlet_entry.political_leaning == 'CENTER':
            bar_colors.append('tab:green')
        elif first_outlet_entry.political_leaning == 'LEFT':
            bar_colors.append('tab:blue')
        elif first_outlet_entry.political_leaning == 'RIGHT':
            bar_colors.append('tab:red')


    return percentages, counts, bar_labels, bar_colors, all_outlets


def plotStats(percentages, counts, bar_labels, bar_colors, title, outlets):
    fig, ax = plt.subplots()
    articles = outlets
    bar_container = ax.bar(articles, percentages, color=bar_colors)
    ax.bar_label(bar_container, fmt='{:,.2f}')

    ax.set_ylabel('Percentage in (%)')
    ax.set_title(title)


    fig2, ax2 = plt.subplots()

    bar_container = ax2.bar(articles, counts, color=bar_colors)
    ax2.bar_label(bar_container, fmt='{:,.0f}')

    ax2.set_ylabel('Count')
    ax2.set_title(title)

    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.setp(ax2.get_xticklabels(), rotation=30, horizontalalignment='right')

    plt.show()


title = 'Count of Articles per Outlet - Training Set'
val_df = load(dirname + '/../../data/processed/training_set_s')
val_df = val_df.loc[val_df["political_leaning"]!="UNDEFINED"]
# outlets = ['NBC News', 'ABC News', 'Fox News', 'NPR', 'Los Angeles Times', 'The Guardian', 'USA Today','Breitbart','HuffPost', 'Slate', 'CBS News', 'The New York Times', 'Reuters']
percentages, counts, bar_labels, bar_colors, outlets = prepareLabels(val_df)
plotStats(percentages, counts, bar_labels, bar_colors, title, outlets)

