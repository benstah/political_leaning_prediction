import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm


class Trainer:
    def train(model, train_dataloader, val_dataloader, learning_rate, epochs):
        best_val_loss = float('inf')
        best_val_acc = float(0)
        early_stopping_threshold_count = 0
        

        # for m1 to use gpu
        use_mps = torch.backends.mps.is_available()
        device = torch.device("mps" if use_mps else "cpu")


        # uncomment for computers which are running on intel
        # use_cuda = torch.cuda.is_available()
        # device = torch.device("cuda" if use_cuda else "cpu")

        # CrossEntropyLoss already implements log_softmax
        criterion = nn.CrossEntropyLoss()
        # TODO: check other values of weight_decay as well
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)

        model = model.to(device)
        criterion = criterion.to(device)

        for epoch in range(epochs):
            total_acc_train = 0
            total_loss_train = 0
            
            model.train()

            for train_input, train_label in tqdm(train_dataloader):
                
                attention_mask = train_input['attention_mask'].to(device)
                input_ids = train_input['input_ids'].squeeze(1).to(device)

                # attention_mask = torch.as_tensor(train_input['attention_mask'], dtype=torch.long).to(device, dtype = torch.long)
                # input_ids = torch.as_tensor(train_input['input_ids'], dtype=torch.long).to(device, dtype = torch.long)

                train_label = torch.as_tensor(train_label).to(device)
                # train_label = train_label.to(device)

                output = model(input_ids, attention_mask)
                # print(output)
                # print(train_label.float().unsqueeze(1))

                # loss = criterion(train_label.float().unsqueeze(1), output)
                train_label = train_label.squeeze_()
                loss = criterion(output, train_label)
                # loss = criterion(output, train_label.float().unsqueeze(1))

                total_loss_train += loss.item()

                acc = ((output >= 0.5).int() == train_label).sum().item()
                # acc = ((output >= 0.5).int() == train_label.unsqueeze(1)).sum().item()
                total_acc_train += acc

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                total_acc_val = 0
                total_loss_val = 0
                
                model.eval()
                
                for val_input, val_label in tqdm(val_dataloader):
                
                    attention_mask = val_input['attention_mask'].to(device)
                    input_ids = val_input['input_ids'].squeeze(1).to(device)

                    val_label = val_label.to(device)

                    output = model(input_ids, attention_mask)

                    val_label = val_label.squeeze_()
                    loss = criterion(output, val_label)
                    # loss = criterion(output, val_label.float().unsqueeze(1))

                    total_loss_val += loss.item()

                    # TODO: check unsqueeze: val_label should be enough to calc accuracy
                    acc = ((output >= 0.5).int() == val_label).sum().item()
                    # acc = ((output >= 0.5).int() == val_label.unsqueeze(1)).sum().item()
                    total_acc_val += acc
                
                print(f'Epochs: {epoch + 1} '
                    f'| Train Loss: {total_loss_train / len(train_dataloader): .3f} '
                    f'| Train Accuracy: {total_acc_train / (len(train_dataloader.dataset)): .3f} '
                    f'| Val Loss: {total_loss_val / len(val_dataloader): .3f} '
                    f'| Val Accuracy: {total_acc_val / len(val_dataloader.dataset): .3f}')

                if best_val_acc < total_acc_val:
                    best_val_acc = total_acc_val
                    torch.save(model, f"best_model.pt")
                    print("Saved model")
                    early_stopping_threshold_count = 0
                else:
                    early_stopping_threshold_count += 1
                
                # look at optimizing early stopping again
                if early_stopping_threshold_count >= 2:
                    print("Early stopping")
                    break