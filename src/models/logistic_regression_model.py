from joblib import load
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import os
import sys
from pathlib import Path 
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError: # Already removed
    pass

from dataset.article_dataset import ArticleDataset

dirname = os.path.dirname(__file__)
os.environ["TOKENIZERS_PARALLELISM"] = "true"



if __name__ == "__main__": 

    def plot_top_words(model, feature_names, n_top_words, title):
        fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
        axes = axes.flatten()
        for topic_idx, topic in enumerate(model):
            top_features_ind = topic.argsort()[-n_top_words:]
            top_features = feature_names[top_features_ind]
            weights = topic[top_features_ind]

            ax = axes[topic_idx]
            ax.barh(top_features, weights, height=0.7)
            ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
            ax.tick_params(axis="both", which="major", labelsize=20)
            for i in "top right left".split():
                ax.spines[i].set_visible(False)
            fig.suptitle(title, fontsize=40)

        plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
        plt.show()

    logisticRegr = LogisticRegression()

    #TODO: step 1: load data (can be raw) and labels (Y)
    # make sure that no undefined is loaded
    training_df = load(dirname + '/../../data/processed_vector/training_set_s')
    training_df = training_df.loc[training_df["political_leaning"]!="UNDEFINED"]

    training_df['headline'] = training_df['headline'].fillna('')
    training_df['body'] = training_df['body'].fillna('')

    headlines = training_df.headline.values.tolist()
    bodies = training_df.body.values.tolist()

    y_train = training_df.label_id.values.tolist()

    corpus = [' '.join(item) for item in zip(headlines, bodies)]

    #TODO: step 2: use countVectorizer to transform articles to vectors
    # eventually remove stop words in english and use max_features
    vectorizer = CountVectorizer(max_features=2000, stop_words="english")
    x_train = vectorizer.fit_transform(corpus)

    df_train_feature_names = vectorizer.get_feature_names_out()

    #TODO step 3: 
    # fit logistic regression with x and y
    logisticRegr.fit(x_train, y_train)

    #TODO step 4:
    #load validation data set
    val_df = load(dirname + '/../../data/processed_vector/validation_set')

    val_df['headline'] = val_df['headline'].fillna('')
    val_df['body'] = val_df['body'].fillna('')

    headlines = val_df.headline.values.tolist()
    bodies = val_df.body.values.tolist()

    y_val = val_df.label_id.values.tolist()

    corpus_val = [' '.join(item) for item in zip(headlines, bodies)]
    x_val = vectorizer.fit_transform(corpus_val)

    #TODO step 5:
    #predict for validation data set
    predictions_val = logisticRegr.predict(x_val)

    #TODO step 6:
    #evaluate model performance on val set
    # Use multiple metrics to evaluate the model
    val_accuracy = accuracy_score(y_val, predictions_val)
    val_precision = precision_score(y_val, predictions_val, average='weighted')
    val_recall = recall_score(y_val, predictions_val, average='weighted')
    val_f1 = f1_score(y_val, predictions_val, average='weighted')

    print('-------------validation metrics-------------')
    print(f'Accuracy: {val_accuracy:.4f}')
    print(f'Precision: {val_precision:.4f}')
    print(f'Recall: {val_recall:.4f}')
    print(f'F1-Score: {val_f1:.4f}\n')

    #TODO step 7:
    #load test set
    test_df = load(dirname + '/../../data/processed_vector/test_set')

    test_df['headline'] = test_df['headline'].fillna('')
    test_df['body'] = test_df['body'].fillna('')

    headlines = test_df.headline.values.tolist()
    bodies = test_df.body.values.tolist()

    y_test = test_df.label_id.values.tolist()

    corpus_val = [' '.join(item) for item in zip(headlines, bodies)]
    x_test = vectorizer.fit_transform(corpus_val)

    #TODO step 8:
    #predict for test set
    predictions = logisticRegr.predict(x_test)

    #TODO step 9:
    # measure performance of test set using multiple metrics
    test_accuracy = accuracy_score(y_test, predictions)
    test_precision = precision_score(y_test, predictions, average='weighted')
    test_recall = recall_score(y_test, predictions, average='weighted')
    test_f1 = f1_score(y_test, predictions, average='weighted')

    print('-------------test metrics-------------')
    print(f'Accuracy: {test_accuracy:.4f}')
    print(f'Precision: {test_precision:.4f}')
    print(f'Recall: {test_recall:.4f}')
    print(f'F1-Score: {test_f1:.4f}\n')

    #TODO step 10:
    #Visualizing stuff
    # plot_top_words(x_train, 30, 'Top 30 words in training data')