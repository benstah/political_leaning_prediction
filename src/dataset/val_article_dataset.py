from torch.utils.data import Dataset
import numpy as np
import re
import nltk
import string
import torch

class ValArticleDataset(Dataset):
    def __init__(self, dataframe, tokenizer):

        headlines = dataframe.headline.values.tolist()
        leads = dataframe.lead.values.tolist()
        bodies = dataframe.body.values.tolist()

        self._print_random_samples(headlines)

        concats = [' '.join(item) for item in zip(headlines, leads, bodies)]

        self.texts = [tokenizer(text, padding='max_length',
                                # max_length=150,
                                truncation=True,
                                return_tensors="pt")
                      for text in concats]
                      

        dataframe["rating"] = dataframe["rating"].map(lambda x: self._numerize_labels(x))

        if 'rating' in dataframe:
            classes = dataframe.rating.values.tolist()
            self.labels = classes

    # translate labels to numbers so that trainer can use labels without using onehot
    def _numerize_labels (self, rating):
        # UNDEFINED = 0
        # RIGHT = 1
        # LEFT = 2
        # CENTER = 3
        # Nothing from the above = 4

        # for leaning in rating:
        if rating == "UNDEFINED":
            return 0
        elif rating == "RIGHT":
            return 1
        elif rating == "LEFT":
            return 2
        elif rating == "CENTER":
            return 3
        else:
            return 4


    def _print_random_samples(self, texts):
        np.random.seed(42)
        random_entries = np.random.randint(0, len(texts), 5)

        for i in random_entries:
            print(f"Entry {i}: {texts[i]}")

        print()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        label = -1
        if hasattr(self, 'labels'):
            label = self.labels[idx]

        return text, label