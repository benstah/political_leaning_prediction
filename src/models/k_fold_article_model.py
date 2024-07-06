from joblib import load
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, DistilBertModel, DistilBertForSequenceClassification, DistilBertForTokenClassification
from sklearn.model_selection import KFold
import math

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

from trainer.article_fold_trainer import KTrainer as t
from dataset.article_dataset import ArticleDataset

dirname = os.path.dirname(__file__)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def reset_weights(m):
    '''
        Try resetting model weights to avoid
        weight leakage.
    '''
    # print ("reset weights")
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            # print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


if __name__ == "__main__": 
    torch.manual_seed(0)
    np.random.seed(0)
        
    BERT_MODEL = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    model = DistilBertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=3)
    val_acc = float(0)

    # TODO: Change for final run. Only for testing performance
    epochs = 1

    # For fold results
    results = {}

    # Load datasets training and val data
    train_df = load(dirname + '/../../data/processed/training_set_s')

    # Only for test purposes, needs to be rmoved for real training loop
    train_df = train_df.loc[train_df["political_leaning"]!="UNDEFINED"]

    train_df = train_df.sample(n=200)

    # Drop other unessary columns
    train_df = train_df.drop(['date_publish', 'authors', 'domain', 'url', 'rating'], axis=1)

    batch_size = 16

    all_outlets = ['Los Angeles Times','NBC News','ABC News','Fox News','NPR','The Guardian','USA Today','Breitbart','HuffPost','Reuters','Slate','CBS News','The New York Times']

    for idx, outlet in enumerate(all_outlets):

        train_data = train_df.loc[train_df["outlet"]!=outlet]
        val_data = train_df.loc[train_df["outlet"]==outlet]

        # Pin memory was initially set to true
        train_dataloader = DataLoader(ArticleDataset(train_data, tokenizer), batch_size=batch_size, num_workers=2, pin_memory=False)
        val_dataloader = DataLoader(ArticleDataset(val_data, tokenizer), batch_size=batch_size, num_workers=2, pin_memory=False)
        # reset model
        print('reset weights')
        model.apply(reset_weights)

    # Recommendations for fine tuning of Bert authors
    # Batch size: 16, 32
    # Learning rate (Adam): 5e-5, 3e-5, 2e-5
    # Number of epochs: 2, 3, 4
        learning_rate = 5e-5
        val_acc = t.train(model, train_dataloader, val_dataloader, learning_rate, epochs, 20, val_acc, len(train_data), len(val_data))
        print ("\ncurrent best val_acc: " +  str(val_acc))
        
        #TODO: remove for final run
        # sys.exit()