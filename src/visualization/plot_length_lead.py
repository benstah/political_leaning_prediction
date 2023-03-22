from joblib import load
import matplotlib.pyplot as plt

import os

dirname = os.path.dirname(__file__)

def prepareLabels(df):
    leads = df.lead.values.tolist()

    article_count = {10: 0, 50: 0, 100: 0, 200:0, 201:0}
    article_percentage = {10: 0.0, 50: 0.0, 100: 0.0, 200:0.0, 201:0.0}
    count_total = len(leads)

    for text in leads:
        length = len(text)

        if length < 10:
            article_count[10] += 1
        elif length < 50:
            article_count[50] += 1
        elif length < 100:
            article_count[100] += 1
        elif length < 200:
            article_count[200] += 1
        elif length >= 201:
            article_count[201] += 1

    percentages = []
    counts = []
    bar_labels = []
    bar_colors = []

    for key in article_count:
        article_percentage[key] = article_count[key] / count_total * 100
        percentages.append(article_percentage[key])
        counts.append(article_count[key])
        bar_labels.append('blue')
        bar_colors.append('tab:blue')
        
    return percentages, counts, bar_labels, bar_colors


def plotStats(percentages, counts, bar_labels, bar_colors, title):
    fig, ax = plt.subplots()
    articles = ['<10', '<50', '<100', '<200', '>=200']
    bar_container = ax.bar(articles, percentages, color=bar_colors)
    ax.bar_label(bar_container, fmt='{:,.2f}')

    ax.set_ylabel('Percentage in %')
    ax.set_title(title)


    fig2, ax2 = plt.subplots()

    bar_container = ax2.bar(articles, counts, color=bar_colors)
    ax2.bar_label(bar_container, fmt='{:,.0f}')

    ax2.set_ylabel('Characters')
    ax2.set_title(title)

    plt.show()


title = 'Length of Leads - Training Set S'
val_df = load(dirname + '/../../data/processed/training_set_s')
percentages, counts, bar_labels, bar_colors = prepareLabels(val_df)
plotStats(percentages, counts, bar_labels, bar_colors, title)


title = 'Length of Leads - Validation Set L'
val_df = load(dirname + '/../../data/processed/training_set_l')
percentages, counts, bar_labels, bar_colors = prepareLabels(val_df)
plotStats(percentages, counts, bar_labels, bar_colors, title)


title = 'Length of Leads - Validation Set'
val_df = load(dirname + '/../../data/processed/validation_set')
percentages, counts, bar_labels, bar_colors = prepareLabels(val_df)
plotStats(percentages, counts, bar_labels, bar_colors, title)


title = 'Length of Leads - Test Set'
val_df = load(dirname + '/../../data/processed/test_set')
percentages, counts, bar_labels, bar_colors = prepareLabels(val_df)
plotStats(percentages, counts, bar_labels, bar_colors, title)




