from joblib import load
import os
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
# from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, DistilBertModel

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
from dataset.article_dataset import ArticleDataset

dirname = os.path.dirname(__file__)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

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

if __name__ == "__main__": 
    torch.manual_seed(0)
    np.random.seed(0)
        
    # eventually use distilBERT  
    # BERT_MODEL = "roberta-base"
    BERT_MODEL = 'distilbert-base-uncased'
    # tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    # base_model = AutoModel.from_pretrained(BERT_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    base_model = DistilBertModel.from_pretrained(BERT_MODEL)

    # TODO: Load datasets training and val data
    train_df = load(dirname + '/../../data/processed/training_set_s')
    val_df = load(dirname + '/../../data/processed/validation_set')

    # Only for test purposes, needs to be rmoved for real training loop
    # TODO: Get statistics for undefined
    train_df = train_df.loc[train_df["political_leaning"]!="UNDEFINED"]
    # train_df = train_df.head(200)
    train_df = train_df.sample(n=200)

    # Drop other unessary columns
    train_df = train_df.drop(['date_publish', 'outlet', 'authors', 'domain', 'url', 'rating'], axis=1)
    # drop political rating, because rating should be the columd for labels
    val_df = val_df.drop(['date_publish', 'outlet', 'authors', 'domain', 'url', 'political_leaning'], axis=1)

    batch_size = 8

    train_dataloader = DataLoader(ArticleDataset(train_df, tokenizer), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dataloader = DataLoader(ArticleDataset(val_df, tokenizer), batch_size=batch_size, num_workers=2, pin_memory=True)

    model = ArticleClassifier(base_model, batch_size)

# Recommendations for fine tuning of Bert authors
# Batch size: 16, 32
# Learning rate (Adam): 5e-5, 3e-5, 2e-5
# Number of epochs: 2, 3, 4
    learning_rate = 1e-4
    epochs = 3
    t.train(model, train_dataloader, val_dataloader, learning_rate, epochs)