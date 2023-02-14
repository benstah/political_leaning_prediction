import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm


class Trainer:
    def train(model, train_dataloader, val_dataloader, learning_rate, epochs):
        best_val_loss = float('inf')
        early_stopping_threshold_count = 0
        
        
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        criterion = nn.BCELoss()
        optimizer = Adam(model.parameters(), lr=learning_rate)

        model = model.to(device)
        criterion = criterion.to(device)

        for epoch in range(epochs):
            total_acc_train = 0
            total_loss_train = 0
            
            model.train()
            
            for train_input, train_label in tqdm(train_dataloader):
                attention_mask = train_input['attention_mask'].to(device)
                input_ids = train_input['input_ids'].squeeze(1).to(device)

                train_label = train_label.to(device)

                output = model(input_ids, attention_mask)

                loss = criterion(output, train_label.float().unsqueeze(1))

                total_loss_train += loss.item()

                acc = ((output >= 0.5).int() == train_label.unsqueeze(1)).sum().item()
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

                    loss = criterion(output, val_label.float().unsqueeze(1))

                    total_loss_val += loss.item()

                    acc = ((output >= 0.5).int() == val_label.unsqueeze(1)).sum().item()
                    total_acc_val += acc
                
                print(f'Epochs: {epoch + 1} '
                    f'| Train Loss: {total_loss_train / len(train_dataloader): .3f} '
                    f'| Train Accuracy: {total_acc_train / (len(train_dataloader.dataset)): .3f} '
                    f'| Val Loss: {total_loss_val / len(val_dataloader): .3f} '
                    f'| Val Accuracy: {total_acc_val / len(val_dataloader.dataset): .3f}')
                
                if best_val_loss > total_loss_val:
                    best_val_loss = total_loss_val
                    torch.save(model, f"best_model.pt")
                    print("Saved model")
                    early_stopping_threshold_count = 0
                else:
                    early_stopping_threshold_count += 1
                    
                if early_stopping_threshold_count >= 1:
                    print("Early stopping")
                    break