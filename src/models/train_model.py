from joblib import load
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, DistilBertModel, DistilBertForSequenceClassification, DistilBertForTokenClassification

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

from trainer.trainer import Trainer as t
from dataset.article_dataset import ArticleDataset

dirname = os.path.dirname(__file__)
os.environ["TOKENIZERS_PARALLELISM"] = "true"


if __name__ == "__main__": 
    torch.manual_seed(0)
    np.random.seed(0)
        
    BERT_MODEL = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    model = DistilBertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=4)

    # Load datasets training and val data
    train_df = load(dirname + '/../../data/processed/training_set_s')
    val_df = load(dirname + '/../../data/processed/validation_set')

    # Only for test purposes, needs to be rmoved for real training loop
    train_df = train_df.loc[train_df["political_leaning"]!="UNDEFINED"]
    train_df = train_df.sample(n=2000)

    # Drop other unessary columns
    train_df = train_df.drop(['date_publish', 'outlet', 'authors', 'domain', 'url', 'rating'], axis=1)
    # drop political rating, because rating should be the column for labels
    val_df = val_df.drop(['date_publish', 'outlet', 'authors', 'domain', 'url', 'political_leaning'], axis=1)

    batch_size = 8

    train_dataloader = DataLoader(ArticleDataset(train_df, tokenizer), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dataloader = DataLoader(ArticleDataset(val_df, tokenizer), batch_size=batch_size, num_workers=2, pin_memory=True)

# Recommendations for fine tuning of Bert authors
# Batch size: 16, 32
# Learning rate (Adam): 5e-5, 3e-5, 2e-5
# Number of epochs: 2, 3, 4
    learning_rate = 1e-5
    epochs = 10
    t.train(model, train_dataloader, val_dataloader, learning_rate, epochs)