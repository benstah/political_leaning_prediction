import numpy
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm


class Trainer:

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

    def printScores(class_name, pres, rec, f1, support, true_positives):

        print ("{:<15} {:<15} {:<15} {:<15} {:<15} {:<10}"
            .format(f'{class_name}: ', 
                    f'| Presicion: {pres: .3f} ',
                    f'| Recall: {rec: .3f} ',
                    f'| f1-score: {f1: .3f} ',
                    f'| Support: {support}',
                    f'| True Positives: {true_positives}',
                    ))

        # print(f'{class_name}: '
        #     f'| Presicion: {pres: .3f} '
        #     f'| Recall: {rec: .3f} '
        #     f'| f1-score: {f1: .3f£} '
        #     f'| Support: {support}')

    def calculateAndDisplayF1Score(class_1, class_2, class_3, class_0 = None):

        total_support = 0
        total_f1 = 0

        if class_0 is not None:
            pres_0, rec_0 = Trainer.getPresicionAndRecall(class_0)
            f1_0 = Trainer.getF1Score(pres_0, rec_0)
            Trainer.printScores("Right 0", pres_0, rec_0, f1_0, class_0["support"], class_0["true_positives"])

            total_f1 = total_f1 + (f1_0 * class_0["support"])
            total_support += class_0["support"]


        pres_1, rec_1 = Trainer.getPresicionAndRecall(class_1)
        f1_1 = Trainer.getF1Score(pres_1, rec_1)
        Trainer.printScores("Left 1", pres_1, rec_1, f1_1, class_1["support"], class_1["true_positives"])

        total_f1 = total_f1 + (f1_1 * class_1["support"])
        total_support += class_1["support"]

        pres_2, rec_2 = Trainer.getPresicionAndRecall(class_2)
        f1_2 = Trainer.getF1Score(pres_2, rec_2)
        Trainer.printScores("Center 3", pres_2, rec_2, f1_2, class_2["support"], class_2["true_positives"])

        total_f1 = total_f1 + (f1_2 * class_2["support"])
        total_support += class_2["support"]

        pres_3, rec_3 = Trainer.getPresicionAndRecall(class_3)
        f1_3 = Trainer.getF1Score(pres_3, rec_3)
        Trainer.printScores("Undefined 4", pres_3, rec_3, f1_3, class_3["support"], class_3["true_positives"])

        total_f1 = total_f1 + (f1_3 * class_3["support"])
        total_support += class_3["support"]

        accuracy_f1 = total_f1 / total_support 
        print(f'Accuray f1: {accuracy_f1}')


    def calculateMoreExposedWeights(min_weight, max_weight, train_weights):
        # min_weight = 50%
        # max_weight = 100%

        # max_weight - min_weight = 50%
        # max_weight - min_weight = 0.03
        # max_weight - weight = 0.015
        # 0.015 / 0.03 = 0.5
        # weight - 0.5 * 0.5 = 0.67

        max_difference = max_weight - min_weight
        actual_difference = max_weight - train_weights
        temp_weight = actual_difference / max_difference

        return train_weights - temp_weight * 0.5
            

    def train(model, train_dataloader, val_dataloader, learning_rate, epochs, model_name, min_weight, max_weight):
        best_val_loss = float('inf')
        best_val_acc = float(0)
        early_stopping_threshold_count = 0
        

        # for m1 to use gpu
        # use_mps = torch.backends.mps.is_available()
        # device = torch.device("mps" if use_mps else "cpu")


        # uncomment for computers which are running on intel
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        torch.cuda.empty_cache()

        # CrossEntropyLoss already implements log_softmax
        # criterion = nn.CrossEntropyLoss()

        # best used 0.0001
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)

        model = model.to(device)
        # criterion = criterion.to(device)

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
            index = 0
            for train_input, train_label, train_article_ids, train_weights in tqdm(train_dataloader):
                
                attention_mask = train_input['attention_mask'].to(device)
                input_ids = train_input['input_ids'].squeeze(1).to(device)

                train_label = torch.as_tensor(train_label).to(device).squeeze_()

                output = model(input_ids, attention_mask=attention_mask, labels=train_label)

                # big problem with the requires_grad -> 
                # The requires grad changed the tensor into the wrong format. Because of CrossEntropyLoss which makes use of log-softmax and NLLLoss
                # I need to make sure that I keep my data in the right format with a grad_fn=<NllLossBackward0>
                # The requieres_grad_() changes the gradient function to grad_fn=requires_grad
                #
                # Example:
                # train_weights = torch.as_tensor(train_weights, dtype=torch.float32).to(device) # .requires_grad_()
                # 
                # If I want to multiply the weight factor with the loss of the loss function, I need to make sure that I don't loose the gradient computation history
                # Therefore instead of writing the criterion to the device with a "mean" reduction, it will not be initialized to the device and with a reduction "none"
                # 
                # To ensure that I don't loose the gradient computation history I need to avoid detaching the logits and use the raw once for my model


                # use the raw logits from the model output
                logits = output.logits

                criterion = nn.CrossEntropyLoss(reduction='none')

                # calculate the loss
                loss = criterion(logits, train_label)
                train_weights = Trainer.calculateMoreExposedWeights(min_weight, max_weight, train_weights)
                train_weights = torch.as_tensor(train_weights, dtype=torch.float32).to(device)

                loss = (loss * train_weights).mean()

                total_loss_train += loss.item()

                preds = output.logits.detach()
                acc = ((preds.argmax(axis=1) == train_label)).sum().item()
                
                pred_labels = preds.argmax(axis=1).cpu().numpy()
                real_labels = train_label.cpu().numpy()

                class_0, class_1, class_2, class_3 = Trainer.countScores(real_labels, pred_labels, class_1, class_2, class_3, class_0)

                index += 1

                total_acc_train += acc

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                total_acc_val = 0
                total_loss_val = 0

                val_class_0 = {
                    "true_positives": 0, 
                    "false_positives": 0,
                    "false_negatives": 0,
                    "support": 0,
                }

                val_class_1 = {
                    "true_positives": 0, 
                    "false_positives": 0,
                    "false_negatives": 0,
                    "support": 0,
                }

                val_class_2 = {
                    "true_positives": 0, 
                    "false_positives": 0,
                    "false_negatives": 0,
                    "support": 0,
                }

                val_class_3 = {
                    "true_positives": 0, 
                    "false_positives": 0,
                    "false_negatives": 0,
                    "support": 0,
                }
                
                model.eval()
                
                for val_input, val_label, val_article_ids in tqdm(val_dataloader):
                
                    attention_mask = val_input['attention_mask'].to(device)
                    input_ids = val_input['input_ids'].squeeze(1).to(device)
                    val_label = torch.as_tensor(val_label).to(device).squeeze_()

                    output = model(input_ids, attention_mask=attention_mask, labels=val_label)

                    criterion = nn.CrossEntropyLoss()
                    loss = criterion(output.logits, val_label)

                    total_loss_val += loss.item()

                    preds = output.logits.detach()
                    # acc = ((preds.argmax(axis=1) >= 0.5).int() == val_label.unsqueeze(1)).sum().item()
                    acc = (preds.argmax(axis=1) == val_label).sum().item()

                    val_pred_labels = preds.argmax(axis=1).cpu().numpy()
                    val_real_labels = val_label.cpu().numpy()

                    val_class_0, val_class_1, val_class_2, val_class_3 = Trainer.countScores(val_real_labels, val_pred_labels, val_class_1, val_class_2, val_class_3, val_class_0)

                    total_acc_val += acc

                
                print(f'Epochs: {epoch + 1} '
                    f'| Train Loss: {total_loss_train / len(train_dataloader): .3f} '
                    f'| Train Accuracy: {total_acc_train / (len(train_dataloader.dataset)): .3f} '
                    f'| Val Loss: {total_loss_val / len(val_dataloader): .3f} '
                    f'| Val Accuracy: {total_acc_val / len(val_dataloader.dataset): .3f}')
                
                print('\n')
                print('------------------------ training scores ---------------------------')
                Trainer.calculateAndDisplayF1Score(class_1, class_2, class_3, class_0)
                
                print('\n')
                print('------------------------ validation scores ---------------------------')
                Trainer.calculateAndDisplayF1Score(val_class_1, val_class_2, val_class_3, val_class_0)

                if best_val_acc < total_acc_val:
                    best_val_acc = total_acc_val
                    torch.save(model, f"" + str(model_name))
                    print("Saved model")
                    early_stopping_threshold_count = 0
                else:
                    early_stopping_threshold_count += 1
                
                if early_stopping_threshold_count >= 2:
                    print("Early stopping")
                    break

