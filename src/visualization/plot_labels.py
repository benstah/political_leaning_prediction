from joblib import load
import matplotlib.pyplot as plt

import os

dirname = os.path.dirname(__file__)

# if political_leaning == "UNDEFINED":
#     return 0
# elif political_leaning == "RIGHT":
#     return 1
# elif political_leaning == "LEFT":
#     return 2
# elif political_leaning == "CENTER":
#     return 3
# else:
#     return 4

def prepareLabels(df):
    label_id = df.label_id.values.tolist()

    article_count = {'left': 0, 'right': 0, 'center': 0, 'undefined':0}
    article_percentage = {'left': 0.0, 'right': 0.0, 'center': 0.0, 'undefined':0.0}
    count_total = len(label_id)

    for id in label_id:

        if id == 0:
            article_count['undefined'] += 1
        elif id == 1:
            article_count['right'] += 1
        elif id == 2:
            article_count['left'] += 1
        elif id == 3:
            article_count['center'] += 1

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
    articles = ['Left', 'Right', 'Center', 'Undefined']
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


title = 'Labels - Training Set S'
val_df = load(dirname + '/../../data/processed/training_set_s')
percentages, counts, bar_labels, bar_colors = prepareLabels(val_df)
plotStats(percentages, counts, bar_labels, bar_colors, title)


title = 'Labels - Validation Set L'
val_df = load(dirname + '/../../data/processed/training_set_l')
percentages, counts, bar_labels, bar_colors = prepareLabels(val_df)
plotStats(percentages, counts, bar_labels, bar_colors, title)


title = 'Labels - Validation Set'
val_df = load(dirname + '/../../data/processed/validation_set')
percentages, counts, bar_labels, bar_colors = prepareLabels(val_df)
plotStats(percentages, counts, bar_labels, bar_colors, title)


title = 'Labels - Test Set'
val_df = load(dirname + '/../../data/processed/test_set')
percentages, counts, bar_labels, bar_colors = prepareLabels(val_df)
plotStats(percentages, counts, bar_labels, bar_colors, title)




