import numpy
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
from joblib import load, dump
import pandas as pd
import os


class KTrainer:

    def countScores(real_labels, pred_labels, class_1, class_2, class_3, class_0 = None):

        # print(type(real_labels))
        # real_labels = numpy.array(list(real_labels.tuple()))
        if numpy.isscalar(real_labels):
            if class_0 is None:
                return class_1, class_2, class_3

            return class_0, class_1, class_2, class_3
        
        if real_labels.size == 0:
            if class_0 is None:
                return class_1, class_2, class_3

            return class_0, class_1, class_2, class_3
        
        if real_labels.size == 1:
            real_labels = real_labels.item()

        if isinstance(real_labels, numpy.ndarray):
        
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

        else:
            print(type(real_labels))

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
            pres_0, rec_0 = KTrainer.getPresicionAndRecall(class_0)
            f1_0 = KTrainer.getF1Score(pres_0, rec_0)
            KTrainer.printScores("Right 0", pres_0, rec_0, f1_0, class_0["support"])

            total_f1 = total_f1 + (f1_0 * class_0["support"])
            total_support += class_0["support"]


        pres_1, rec_1 = KTrainer.getPresicionAndRecall(class_1)
        f1_1 = KTrainer.getF1Score(pres_1, rec_1)
        KTrainer.printScores("Left 1", pres_1, rec_1, f1_1, class_1["support"])

        total_f1 = total_f1 + (f1_1 * class_1["support"])
        total_support += class_1["support"]

        pres_2, rec_2 = KTrainer.getPresicionAndRecall(class_2)
        f1_2 = KTrainer.getF1Score(pres_2, rec_2)
        KTrainer.printScores("Center 3", pres_2, rec_2, f1_2, class_2["support"])

        total_f1 = total_f1 + (f1_2 * class_2["support"])
        total_support += class_2["support"]

        pres_3, rec_3 = KTrainer.getPresicionAndRecall(class_3)
        f1_3 = KTrainer.getF1Score(pres_3, rec_3)
        KTrainer.printScores("Undefined 4", pres_3, rec_3, f1_3, class_3["support"])

        total_f1 = total_f1 + (f1_3 * class_3["support"])
        total_support += class_3["support"]

        accuracy_f1 = total_f1 / total_support 
        print(f'Accuray f1: {accuracy_f1}')

    def save_scores_to_dataset(val_score_list, column_name):

        print("saving to dataset:" + column_name + "\n")
        # print(val_score_list)
        
        # load dataset
        dirname = os.path.dirname(__file__)
        filename = dirname + '/../../data/processed/training_set_s'
        df = load(filename)
        # update row where id in map
        
        for id, val_score in val_score_list.items():
            df.loc[df['id'] == id, column_name] = val_score

        # save dataset
        dump(df, filename, compress=4)
            

    def train(model, train_dataloader, val_dataloader, learning_rate, epochs, iteration, val_acc=None, len_train=None, len_val=None):
    
        val_score_ids_list = {}

        best_val_loss = float('inf')
        
        best_val_acc = float(0)
        early_stopping_threshold_count = 0

        kfold_best_val = float(0)
        

        # for m1 to use gpu
        # use_mps = torch.backends.mps.is_available()
        # device = torch.device("mps" if use_mps else "cpu")


        # uncomment for computers which are running on intel
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        torch.cuda.empty_cache()

        # CrossEntropyLoss already implements log_softmax
        criterion = nn.CrossEntropyLoss()

        # TODO: check other values of weight_decay as well
        # TODO: Try lr=3e-5 and weight_decay=0.3
        # TODO: Try 1e-4, 1e-3, 1e-2, 1e-1
        # best so far 0.0001
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)

        model = model.to(device)
        criterion = criterion.to(device)

        for epoch in range(epochs):
            total_acc_train = 0
            total_loss_train = 0

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

            
            model.train()
            index = 1
            for train_input, train_label, train_article_ids in tqdm(train_dataloader):
                
                attention_mask = train_input['attention_mask'].to(device)
                input_ids = train_input['input_ids'].squeeze(1).to(device)

                train_label = torch.as_tensor(train_label).to(device).squeeze_()

                output = model(input_ids, attention_mask=attention_mask, labels=train_label)

                loss = output.loss

                total_loss_train += loss.item()

                preds = output.logits.detach()
                acc = ((preds.argmax(axis=1) == train_label)).sum().item()
                
                pred_labels = preds.argmax(axis=1).cpu().squeeze().numpy()
                real_labels = train_label.cpu().squeeze().numpy()

                class_0, class_1, class_2, class_3 = KTrainer.countScores(real_labels, pred_labels, class_1, class_2, class_3, class_0)

                # if index == 0 or index == 50 or index == 100 or index == 150 or index == 200:
                #     print(class_0)
                #     print(class_1)
                #     print(class_2)
                #     print(class_3)

                index += 1

                total_acc_train += acc

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                total_acc_val = 0
                total_loss_val = 0
                
                model.eval()
                
                for val_input, val_label, val_article_ids in tqdm(val_dataloader):
                
                    attention_mask = val_input['attention_mask'].to(device)
                    input_ids = val_input['input_ids'].squeeze(1).to(device)
                    val_label = torch.as_tensor(val_label).to(device).squeeze_()

                    output = model(input_ids, attention_mask=attention_mask, labels=val_label)

                    loss = criterion(output.logits, val_label)

                    total_loss_val += loss.item()

                    preds = output.logits.detach()
                    # acc = ((preds.argmax(axis=1) >= 0.5).int() == val_label.unsqueeze(1)).sum().item()
                    acc = (preds.argmax(axis=1) == val_label).sum().item()

                    total_acc_val += acc

                    # len_train = (len(train_dataloader.dataset) / k_folds) * (k_folds - 1)
                    # len_acc = len(train_dataloader.dataset) / k_folds

                    # collect all ids of the validationset
                    for val_id in val_article_ids.tolist():
                        val_score_ids_list[val_id] = 0.0

                
                print(f'Epochs: {epoch + 1} '
                    f'| Train Loss: {total_loss_train / len(train_dataloader): .3f} '
                    f'| Train Accuracy: {total_acc_train / len_train: .3f} '
                    f'| Val Loss: {total_loss_val / len(val_dataloader): .3f} '
                    f'| Val Accuracy: {total_acc_val / len_val: .3f}')
                
                print('\n')
                KTrainer.calculateAndDisplayF1Score(class_1, class_2, class_3, class_0)

                if best_val_acc < total_acc_val:
                    best_val_acc = total_acc_val
                    # torch.save(model, f"best_model.pt")
                    print("Saved model")
                    early_stopping_threshold_count = 0

                    # update dataset
                    # add val score to weight variable of dataset
                    for id, score in val_score_ids_list.items():
                        val_score_ids_list[id] = total_acc_val / len_val

                    column = "w" + str(iteration)
                    # for the algorithm, which was running on index 0
                    # column = "w" + str(iteration+1)
                    KTrainer.save_scores_to_dataset(val_score_ids_list, column)
                else:
                    early_stopping_threshold_count += 1


                # print(val_score_ids_list)
                # print("\n---------\n")
                # print(str(len_acc))
                # print("\n---------\n")
                # print(str(kfold_best_val))
                # print("\n---------\n")
                # print(str(total_acc_train / len_acc))

                if kfold_best_val < (total_acc_val / len_val):
                    kfold_best_val = (total_acc_val / len_val)
                    
                
                if early_stopping_threshold_count >= 3:
                    print("Early stopping")
                    break

        

        if val_acc is not None:
            if best_val_acc > val_acc:
                val_acc = best_val_acc

            return val_acc

