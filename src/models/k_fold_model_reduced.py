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

from trainer.k_fold_trainer import KTrainer as t
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


 # Create 4 new columns in dataset 'w1', 'w2', 'w3', 'wa' (weight average)
 # TODO: Upload new dataset to the cloud for downloading it
 # TODO: Wrap first for loop in another loop with 3 iterations
 # TODO: Per iteration safe validation in 'w' per each item in batch
 # TODO: last iteration calculate average and safe in wa
 # TODO: For Prod: split = len / batch_size
 # TODO: For Prod: characters back to 400

if __name__ == "__main__": 

    # started with 2 for the different w columns
    for i in range(4, 10):

        print("--------------- start with iteration " + str(i) + " -------------------------")
    # I used the range because server is constantly blocked and I should run everything all in once so that I don't need to wait for too long
    # Initial plan was to run every iteration manually one by one and caculate average in between to see if further steps would be necesary
    # i = 3
        torch.manual_seed(i)
        np.random.seed(i)
            
        BERT_MODEL = 'distilbert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
        model = DistilBertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=3)
        val_acc = float(0)

        # TODO: Change for final run. Only for testing performance
        epochs = 2

        # For fold results
        results = {}

        # Load datasets training and val data
        train_df = load(dirname + '/../../data/processed/training_set_s')

        # Only for test purposes, needs to be rmoved for real training loop
        train_df = train_df.loc[train_df["political_leaning"]!="UNDEFINED"]

        # this amount of k fold would even on a gpu server run for approx. 2,5 years. Not feasable for this thesis
        # k_folds = math.floor(len(train_df) / 32)

        # new approach: Try to keep the kFolds as small as possible to let the validation score be as least noisy as possible.
        # With this approach the algorithm runs approx. only 2 weeks
        k_folds = 12
        # k_folds = math.floor(len(train_df) / 2048)

        kfold = KFold(n_splits=k_folds, shuffle=True)

        # train_df = train_df.sample(n=200)

        # Drop other unessary columns
        train_df = train_df.drop(['date_publish', 'outlet', 'authors', 'domain', 'url', 'rating'], axis=1)

        batch_size = 16

        for fold, (train_ids, test_ids) in enumerate(kfold.split(train_df)):

            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            # Pin memory was initially set to true
            train_dataloader = DataLoader(ArticleDataset(train_df, tokenizer), batch_size=batch_size, sampler=train_subsampler, num_workers=2, pin_memory=False)
            val_dataloader = DataLoader(ArticleDataset(train_df, tokenizer), batch_size=batch_size, sampler=test_subsampler, num_workers=2, pin_memory=False)

            # reset model
            # print('reset weights')
            model.apply(reset_weights)

        # Recommendations for fine tuning of Bert authors
        # Batch size: 16, 32
        # Learning rate (Adam): 5e-5, 3e-5, 2e-5
        # Number of epochs: 2, 3, 4
            learning_rate = 5e-5
            val_acc = t.train(model, train_dataloader, val_dataloader, learning_rate, epochs, i, val_acc, k_folds)
            print ("\ncurrent best val_acc: " +  str(val_acc))