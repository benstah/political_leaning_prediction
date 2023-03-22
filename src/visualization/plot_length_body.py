from joblib import load
import matplotlib.pyplot as plt

import os

dirname = os.path.dirname(__file__)

def prepareLabels(df):
    bodies = df.body.values.tolist()

    article_count = {500: 0, 1000: 0, 1500: 0, 2000:0, 2500:0, 3000:0, 3500:0, 4000: 0, 4500:0, 5000:0, 5500:0, 5001:0}
    article_percentage = {500: 0.0, 1000: 0.0, 1500: 0.0, 2000:0.0, 2500:0.0, 3000:0.0, 3500:0.0, 4000: 0.0, 4500: 0.0, 5000: 0.0, 5500:0.0, 5001: 0.0}
    count_total = len(bodies)

    for text in bodies:
        length = len(text)

        if length < 500:
            article_count[500] += 1
        elif length < 1000:
            article_count[1000] += 1
        elif length < 1500:
            article_count[1500] += 1
        elif length < 2000:
            article_count[2000] += 1
        elif length < 2500:
            article_count[2500] += 1
        elif length < 3000:
            article_count[3000] += 1
        elif length < 3500:
            article_count[3500] += 1
        elif length < 4000:
            article_count[4000] += 1
        elif length < 4500:
            article_count[4500] += 1
        elif length < 5000:
            article_count[5000] += 1
        elif length < 5500:
            article_count[5500] += 1
        elif length >= 5500:
            article_count[5001] += 1

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
    articles = ['<500', '<1000', '<1500', '<2000', '<2500', '<3000', '<3500', '<4000', '<4500', '<5000', '<5500', '>=5500']
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


title = 'Length of Body - Training Set S'
val_df = load(dirname + '/../../data/processed/training_set_s')
percentages, counts, bar_labels, bar_colors = prepareLabels(val_df)
plotStats(percentages, counts, bar_labels, bar_colors, title)


title = 'Length of Body - Validation Set L'
val_df = load(dirname + '/../../data/processed/training_set_l')
percentages, counts, bar_labels, bar_colors = prepareLabels(val_df)
plotStats(percentages, counts, bar_labels, bar_colors, title)


title = 'Length of Body - Validation Set'
val_df = load(dirname + '/../../data/processed/validation_set')
percentages, counts, bar_labels, bar_colors = prepareLabels(val_df)
plotStats(percentages, counts, bar_labels, bar_colors, title)


title = 'Length of Body - Test Set'
val_df = load(dirname + '/../../data/processed/test_set')
percentages, counts, bar_labels, bar_colors = prepareLabels(val_df)
plotStats(percentages, counts, bar_labels, bar_colors, title)




