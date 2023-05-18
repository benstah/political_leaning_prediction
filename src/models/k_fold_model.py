from joblib import load
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, DistilBertModel, DistilBertForSequenceClassification, DistilBertForTokenClassification
from sklearn.model_selection import KFold

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

from trainer.k_fold_trainer import KTrainer as t
from dataset.article_dataset import ArticleDataset

dirname = os.path.dirname(__file__)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def reset_weights(m):
    '''
        Try resetting model weights to avoid
        weight leakage.
    '''
    print ("reset weights")
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

    k_folds = 5
    epochs = 3

    # For fold results
    results = {}

    # Load datasets training and val data
    train_df = load(dirname + '/../../data/processed/training_set_s')
    val_df = load(dirname + '/../../data/processed/validation_set')

    # Only for test purposes, needs to be rmoved for real training loop
    train_df = train_df.loc[train_df["political_leaning"]!="UNDEFINED"]

    kfold = KFold(n_splits=k_folds, shuffle=True)

    train_df = train_df.sample(n=4000)

    # Drop other unessary columns
    train_df = train_df.drop(['date_publish', 'outlet', 'authors', 'domain', 'url', 'rating'], axis=1)
    # drop political rating, because rating should be the column for labels
    val_df = val_df.drop(['date_publish', 'outlet', 'authors', 'domain', 'url', 'political_leaning'], axis=1)

    batch_size = 8

    for fold, (train_ids, test_ids) in enumerate(kfold.split(train_df)):

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        train_dataloader = DataLoader(ArticleDataset(train_df, tokenizer), batch_size=batch_size, sampler=train_subsampler, num_workers=2, pin_memory=True)
        val_dataloader = DataLoader(ArticleDataset(train_df, tokenizer), batch_size=batch_size, sampler=test_subsampler, num_workers=2, pin_memory=True)

        # reset model
        model.apply(reset_weights)

    # Recommendations for fine tuning of Bert authors
    # Batch size: 16, 32
    # Learning rate (Adam): 5e-5, 3e-5, 2e-5
    # Number of epochs: 2, 3, 4
        learning_rate = 1e-5
        val_acc = t.train(model, train_dataloader, val_dataloader, learning_rate, epochs, val_acc)
        print ("\ncurrent best val_acc: " +  val_acc)

        # Print fold results
    # print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    # print('--------------------------------')
    # sum = 0.0
    # for key, value in results.items():
    #     print(f'Fold {key}: {value} %')
    #     sum += value
    # print(f'Average: {sum/len(results.items())} %')