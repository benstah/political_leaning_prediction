from torch.utils.data import Dataset
import numpy as np
import re
import nltk
import string
import torch

class ArticleDataset(Dataset):
    def __init__(self, dataframe, tokenizer):

        headlines = dataframe.headline.values.tolist()
        bodies = dataframe.body.values.tolist()

        concats = [' '.join(item) for item in zip(headlines, bodies)]

        self._print_random_samples(headlines)

        # truncate texts to 5000 characters
        # all_texts = [text[:4500] if len(text) > 4500 else text for text in concats]
        
        #TODO: filter out articles that are too long
        #TODO: split article in multiple
        #TODO: max_length
        #TODO: take part of undefined data

        self.texts = [tokenizer(text, padding='max_length',
                                max_length=400,
                                # max_length=150, # only for testing purposes
                                truncation=True,
                                return_tensors="pt")
                      # for text in all_texts]
                        for text in concats]

       #  self.texts = [tokenizer(text, add_special_tokens=True) for text in all_texts]
        
        if 'label_id' in dataframe:
            classes = dataframe.label_id.values.tolist()
            self.labels = classes

        if "id" in dataframe:
            ids = dataframe.id.values.tolist()
            self.ids = ids


    def _print_random_samples(self, texts):
        np.random.seed(42)
        random_entries = np.random.randint(0, len(texts), 5)

        for i in random_entries:
            print(f"Entry {i}: {texts[i]}")


    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        label = -1
        if hasattr(self, 'labels'):
            label = self.labels[idx]

        if hasattr(self, 'ids'):
            id = self.ids[idx]

        return text, label, id