from joblib import load
import matplotlib.pyplot as plt

import os

dirname = os.path.dirname(__file__)

def prepareLabels(df):
    headlines = df.headline.values.tolist()
    leads = df.lead.values.tolist()
    bodies = df.body.values.tolist()

    concats = [' '.join(item) for item in zip(headlines, leads, bodies)]

    article_count = {200: 0, 400: 0, 600: 0, 800:0, 1000:0, 1200:0, 1400:0, 1600: 0, 1800:0, 2000:0, 2200:0, 2201:0}
    article_percentage = {200: 0.0, 400: 0.0, 600: 0.0, 800:0.0, 1000:0.0, 1200:0.0, 1400:0.0, 1600: 0.0, 1800: 0.0, 2000: 0.0, 2200:0.0, 2201: 0.0}
    count_total = len(concats)

    for text in concats:
        length = text.count(' ')

        if length < 200:
            article_count[200] += 1
        elif length < 400:
            article_count[400] += 1
        elif length < 600:
            article_count[600] += 1
        elif length < 800:
            article_count[800] += 1
        elif length < 1000:
            article_count[1000] += 1
        elif length < 1200:
            article_count[1200] += 1
        elif length < 1400:
            article_count[1400] += 1
        elif length < 1600:
            article_count[1600] += 1
        elif length < 1800:
            article_count[1800] += 1
        elif length < 2000:
            article_count[2000] += 1
        elif length < 2200:
            article_count[2200] += 1
        elif length >= 2200:
            article_count[2201] += 1

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
    articles = ['<200', '<400', '<600', '<800', '<1000', '<1200', '<1400', '<1600', '<1800', '<2000', '<2200', '>=2200']
    bar_container = ax.bar(articles, percentages, color=bar_colors)
    ax.bar_label(bar_container, fmt='{:,.2f}')

    ax.set_ylabel('Percentage in %')
    ax.set_title(title)


    fig2, ax2 = plt.subplots()

    bar_container = ax2.bar(articles, counts, color=bar_colors)
    ax2.bar_label(bar_container, fmt='{:,.0f}')

    ax2.set_ylabel('Words')
    ax2.set_title(title)

    plt.show()


title = 'Words in Articles (Head, Lead, Body) - Training Set S'
val_df = load(dirname + '/../../data/processed/training_set_s')
percentages, counts, bar_labels, bar_colors = prepareLabels(val_df)
plotStats(percentages, counts, bar_labels, bar_colors, title)


title = 'Words in Articles (Head, Lead, Body) - Training Set L'
val_df = load(dirname + '/../../data/processed/training_set_l')
percentages, counts, bar_labels, bar_colors = prepareLabels(val_df)
plotStats(percentages, counts, bar_labels, bar_colors, title)


title = 'Words in Articles (Head, Lead, Body) - Validation Set'
val_df = load(dirname + '/../../data/processed/validation_set')
percentages, counts, bar_labels, bar_colors = prepareLabels(val_df)
plotStats(percentages, counts, bar_labels, bar_colors, title)


title = 'Words in Articles (Head, Lead, Body) - Test Set'
val_df = load(dirname + '/../../data/processed/test_set')
percentages, counts, bar_labels, bar_colors = prepareLabels(val_df)
plotStats(percentages, counts, bar_labels, bar_colors, title)




