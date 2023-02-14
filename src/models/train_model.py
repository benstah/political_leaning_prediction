from joblib import load
import os
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModel


from ..classifier.article_classifier import ArticleClassifier
from ..trainer.trainer import Trainer as t

dirname = os.path.dirname(__file__)


torch.manual_seed(0)
np.random.seed(0)
    
    
BERT_MODEL = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
base_model = AutoModel.from_pretrained(BERT_MODEL)

# TODO: Load datasets training and val data
train_df = load(dirname + '/../../data/processed/training_set_s')
val_df = load(dirname + '/../../data/processed/validation_set')

# TODO: Make sure right column is called 'labels'

# TODO: Drop other unessary columns

# TODO: tokenize all necessary data


train_dataloader = DataLoader(train_df, batch_size=8, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_df, batch_size=8, num_workers=0)

model = ArticleClassifier(base_model)


learning_rate = 1e-5
epochs = 5
t.train(model, train_dataloader, val_dataloader, learning_rate, epochs)