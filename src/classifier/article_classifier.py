from torch import nn
import torch

class ArticleClassifier(nn.Module):
    def __init__(self, base_model):
        super(ArticleClassifier, self).__init__()

        self.bert = base_model
        self.fc1 = nn.Linear(768, 32) # Notable here is the input size of the first linear layer of 768, which is going to correspond to the hidden layer dimension of our chosen base model for the sequence contextualization.
        self.fc2 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask):
        ids = torch.stack(input_ids)
        # print(ids)
        # print(input_ids)
        bert_out = self.bert(input_ids=input_ids[0], attention_mask=attention_mask[0])[0][:, 0]
        # bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
        x = self.fc1(bert_out)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x