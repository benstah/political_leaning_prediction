from joblib import load
import os
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModel

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

from classifier.article_classifier import ArticleClassifier
from trainer.trainer import Trainer as t
from dataset.train_article_dataset import TrainArticleDataset
from dataset.val_article_dataset import ValArticleDataset

dirname = os.path.dirname(__file__)

# columns that need to be tokenized headline,lead, body
def tokenizeData(df, tokenizer):
    columns = [df.headline, df.lead, df.body]

    for column in columns:
        texts = column.values.tolist()
        texts = [tokenizer(text, padding='max_length',
                                    # max_length=150,
                                    truncation=True,
                                    return_tensors="pt")
                            for text in texts]
        column = texts

    return df

torch.manual_seed(0)
np.random.seed(0)
    
# eventually use distilBERT  
BERT_MODEL = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
base_model = AutoModel.from_pretrained(BERT_MODEL)

# TODO: Load datasets training and val data
train_df = load(dirname + '/../../data/processed/training_set_s')
val_df = load(dirname + '/../../data/processed/validation_set')

# Only for test purposes, needs to be rmoved for real training loop
# TODO: Get statistics for undefined
train_df = train_df[train_df["rating"]!="UNDEFINED"]
train_df = train_df.head(2000)

# TODO: Drop other unessary columns
train_df = train_df.drop(['date_publish', 'outlet', 'authors', 'domain', 'url', 'rating'], axis=1)
# drop political rating, because rating should be the columd for labels
val_df = val_df.drop(['date_publish', 'outlet', 'authors', 'domain', 'url', 'political_leaning'], axis=1)

# TODO: Make sure right column is called 'labels' id;date_publish;outlet;headline;lead;body;authors;domain;url;political_leaning;rating
# train_df.rename(columns={'political_leaning': 'labels'}, inplace=True)
# train_df.rename(columns={'id': 'Id', 'political_leaning': 'label'}, inplace=True)
# val_df.rename(columns={'rating': 'labels'}, inplace=True)
#val_df.rename(columns={'id': 'Id', 'rating': 'label'}, inplace=True)

print(train_df.head)

# TODO: tokenize all necessary data
train_df = tokenizeData(train_df, tokenizer)
val_df = tokenizeData(val_df, tokenizer)

train_dataloader = DataLoader(TrainArticleDataset(train_df, tokenizer), batch_size=16, shuffle=True, num_workers=0)
val_dataloader = DataLoader(ValArticleDataset(val_df, tokenizer), batch_size=16, num_workers=0)
# train_dataloader = DataLoader(train_df, batch_size=8, shuffle=True, num_workers=0)
# val_dataloader = DataLoader(val_df, batch_size=8, num_workers=0)

model = ArticleClassifier(base_model)


learning_rate = 1e-5
epochs = 5
t.train(model, train_dataloader, val_dataloader, learning_rate, epochs)