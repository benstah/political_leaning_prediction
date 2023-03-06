from torch.utils.data import Dataset
import numpy as np
import re
import nltk
import string
import torch

class ArticleDataset(Dataset):
    def __init__(self, dataframe, tokenizer):

        headlines = dataframe.headline.values.tolist()
        leads = dataframe.lead.values.tolist()
        bodies = dataframe.body.values.tolist()


        concats = [' '.join(item) for item in zip(headlines, leads, bodies)]

        self._print_random_samples(headlines)

        self.texts = [tokenizer(text, padding='max_length',
                                # max_length=150,
                                truncation=True,
                                return_tensors="pt")
                      for text in concats]
        
        if 'label_id' in dataframe:
            classes = dataframe.label_id.values.tolist()
            self.labels = classes


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