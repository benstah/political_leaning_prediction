from joblib import load
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, DistilBertModel, DistilBertForSequenceClassification, DistilBertForTokenClassification
import pandas as pd

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

    def countScores(real_labels, pred_labels, class_1, class_2, class_3, class_0 = None):
        
        for i, label in enumerate(real_labels):
            # count support
            if label == 0:
                class_0["support"] += 1
            elif label == 1:
                class_1["support"] += 1
            elif label == 2:
                class_2["support"] += 1
            elif label == 3:
                class_3["support"] += 1

            # true positive
            if label == pred_labels[i]:
                if label == 0:
                    class_0["true_positives"] += 1
                elif label == 1:
                    class_1["true_positives"] += 1
                elif label == 2:
                    class_2["true_positives"] += 1
                elif label == 3:
                    class_3["true_positives"] += 1
            
            # false positives and false negatives
            else:
                # false negatives
                if pred_labels[i] == 0:
                    class_0["false_positives"] += 1
                elif pred_labels[i] == 1:
                    class_1["false_positives"] += 1
                elif pred_labels[i] == 2:
                    class_2["false_positives"] += 1
                elif pred_labels[i] == 3:
                    class_3["false_positives"] += 1

                # false positives
                if label == 0:
                    class_0["false_negatives"] += 1
                elif label == 1:
                    class_1["false_negatives"] += 1
                elif label == 2:
                    class_2["false_negatives"] += 1
                elif label == 3:
                    class_3["false_negatives"] += 1

        if class_0 is None:
            return class_1, class_2, class_3

        return class_0, class_1, class_2, class_3
    

    # precision = true positives / true positives + false positives
    # recall = true positives / true positives + false negatives
    def getPresicionAndRecall(class_counts):
        presicion = 0
        recall = 0

        division_value_p = (class_counts["true_positives"] + class_counts["false_positives"])
        division_value_r = (class_counts["true_positives"] + class_counts["false_negatives"])

        if division_value_p == 0:
            presicion = 0
        else:
            presicion = class_counts["true_positives"] / division_value_p

        if division_value_r == 0:
            recall = 0
        else:
            recall = class_counts["true_positives"] / division_value_r

        return presicion, recall

    # f1 = 2 * (precision * recall / precision + recall )
    def getF1Score(presicion, recall):

        if presicion == 0 and recall == 0:
            return 0
        
        return 2 * ((presicion * recall) / (presicion + recall))

    def printScores(class_name, pres, rec, f1, support):

        print ("{:<15} {:<15} {:<15} {:<15} {:<10}"
            .format(f'{class_name}: ', 
                    f'| Presicion: {pres: .3f} ',
                    f'| Recall: {rec: .3f} ',
                    f'| f1-score: {f1: .3f} ',
                    f'| Support: {support}'
                    ))

        # print(f'{class_name}: '
        #     f'| Presicion: {pres: .3f} '
        #     f'| Recall: {rec: .3f} '
        #     f'| f1-score: {f1: .3f} '
        #     f'| Support: {support}')

    def calculateAndDisplayF1Score(class_1, class_2, class_3, class_0 = None):

        total_support = 0
        total_f1 = 0

        if class_0 is not None:
            pres_0, rec_0 = getPresicionAndRecall(class_0)
            f1_0 = getF1Score(pres_0, rec_0)
            printScores("Right 0", pres_0, rec_0, f1_0, class_0["support"])

            total_f1 = total_f1 + (f1_0 * class_0["support"])
            total_support += class_0["support"]


        pres_1, rec_1 = getPresicionAndRecall(class_1)
        f1_1 = getF1Score(pres_1, rec_1)
        printScores("Left 1", pres_1, rec_1, f1_1, class_1["support"])

        total_f1 = total_f1 + (f1_1 * class_1["support"])
        total_support += class_1["support"]

        pres_2, rec_2 = getPresicionAndRecall(class_2)
        f1_2 = getF1Score(pres_2, rec_2)
        printScores("Center 3", pres_2, rec_2, f1_2, class_2["support"])

        total_f1 = total_f1 + (f1_2 * class_2["support"])
        total_support += class_2["support"]

        pres_3, rec_3 = getPresicionAndRecall(class_3)
        f1_3 = getF1Score(pres_3, rec_3)
        printScores("Undefined 4", pres_3, rec_3, f1_3, class_3["support"])

        total_f1 = total_f1 + (f1_3 * class_3["support"])
        total_support += class_3["support"]

        accuracy_f1 = total_f1 / total_support 
        print(f'Accuray f1: {accuracy_f1}')






    def get_predictions(model, loader):
        # for m1 to use gpu
        # use_mps = torch.backends.mps.is_available()
        # device = torch.device("mps" if use_mps else "cpu")

        class_0 = {
            "true_positives": 0, 
            "false_positives": 0,
            "false_negatives": 0,
            "support": 0,
        }

        class_1 = {
            "true_positives": 0, 
            "false_positives": 0,
            "false_negatives": 0,
            "support": 0,
        }

        class_2 = {
            "true_positives": 0, 
            "false_positives": 0,
            "false_negatives": 0,
            "support": 0,
        }

        class_3 = {
            "true_positives": 0, 
            "false_positives": 0,
            "false_negatives": 0,
            "support": 0,
        }

        # machines running on intel
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        model = model.to(device)
        
        
        results_predictions = []
        with torch.no_grad():
            model.eval()
            acc = 0
            for data_input, test_label, test_label_ids in tqdm(loader):
                attention_mask = data_input['attention_mask'].to(device)
                input_ids = data_input['input_ids'].squeeze(1).to(device)
                test_label = torch.as_tensor(test_label).to(device).squeeze_()

                output = model(input_ids, attention_mask=attention_mask, labels=test_label)
                
                preds = output.logits.detach()
                acc = acc + (preds.argmax(axis=1) == test_label).sum().item()
                results_predictions.append(preds.argmax(axis=1))

                real_labels = test_label.cpu().numpy()
                pred_labels = preds.argmax(axis=1).cpu().numpy()

                class_0, class_1, class_2, class_3 = countScores(real_labels, pred_labels, class_1, class_2, class_3, class_0)
        
        # print(torch.tensor(results_predictions).cpu().detach().numpy())
        accurracy = acc / len(loader.dataset)
        print('Test Accuracy: ' + str(accurracy))
        calculateAndDisplayF1Score(class_1, class_2, class_3, class_0)
        return torch.cat(results_predictions).cpu().detach().numpy()

    model = torch.load("best_model_w1_exposed.pt", map_location='cpu')

    BERT_MODEL = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

    # Load test set
    test_df = load(dirname + '/../../data/processed/test_set')

    # drop political rating, because rating should be the column for labels
    test_df = test_df.drop(['date_publish', 'outlet', 'authors', 'domain', 'url', 'political_leaning'], axis=1)

    batch_size = 16

    test_dataLoader = DataLoader(ArticleDataset(test_df, tokenizer), batch_size=batch_size, num_workers=1, pin_memory=True)

    sample_submission = pd.read_csv(dirname + "/../../weight_one_exposed_submission.csv")

    sample_submission["target"] = get_predictions(model, test_dataLoader)

    sample_submission.to_csv("weight_one_exposed_submission.csv", index=False)
