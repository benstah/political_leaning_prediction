from torch import nn
import torch


class ArticleClassifier(nn.Module):
    def __init__(self, base_model):
        super(ArticleClassifier, self).__init__()

        self.bert = base_model
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 4)

    def forward(self, input_ids, attention_mask):
        output_1 = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
    


# class ArticleClassifier(nn.Module):
#     def __init__(self, base_model):
#         super(ArticleClassifier, self).__init__()

#         in_features=768
#         linear_size = 3

#         self.bert = base_model
#         self.dropout1 = nn.Dropout()
#         self.linear1 = nn.Linear(in_features=in_features, out_features=linear_size)
#         self.batch_norm1 = nn.BatchNorm1d(num_features=linear_size)
#         self.dropout2 = nn.Dropout(p=0.8)
#         self.linear2 = nn.Linear(in_features=linear_size, out_features=1)
#         self.batch_norm2 = nn.BatchNorm1d(num_features=1)
#         self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax(dim=1)
        
#     def forward(self, tokens, attention_mask):
#         bert_output = self.bert(input_ids=tokens, attention_mask=attention_mask)[0][:, 0]
#         x = self.dropout1(bert_output)
#         x = self.linear1(x)
#         x = self.dropout2(x)
#         x = self.batch_norm1(x)
#         x = self.linear2(x)
#         x = self.batch_norm2(x)
#         return x
#         # return self.softmax(x)
    
#     def freeze_bert(self):
#         """
#         Freezes the parameters of BERT so when BertWithCustomNNClassifier is trained
#         only the wieghts of the custom classifier are modified.
#         """
#         for param in self.bert.named_parameters():
#             param[1].requires_grad=False
    
#     def unfreeze_bert(self):
#         """
#         Unfreezes the parameters of BERT so when BertWithCustomNNClassifier is trained
#         both the wieghts of the custom classifier and of the underlying BERT are modified.
#         """
#         for param in self.bert.named_parameters():
#             param[1].requires_grad=True


# class ArticleClassifier(nn.Module):
#     def __init__(self, base_model):
#         super(ArticleClassifier, self).__init__()

#         self.bert = base_model
#         self.fc1 = nn.Linear(768, 1) # Notable here is the input size of the first linear layer of 768, which is going to correspond to the hidden layer dimension of our chosen base model for the sequence contextualization.
#         # self.fc2 = nn.Linear(32, 1)

#         # self.relu = nn.ReLU()
#         # self.softmax = nn.Softmax(dim=1)
#         # self.sigmoid = nn.Sigmoid()
#         self.dropout = nn.Dropout(p=0.3)
        
#     def forward(self, input_ids, attention_mask):
#         # ids = torch.stack(input_ids)
#         # bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
#         bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         print(bert_out)
#         # bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
#         # x = self.fc1(bert_out)
#         # x = self.relu(x)
        
#         # x = self.sigmoid(x)
#         # x = self.fc2(x)
#         # x = self.relu(x)
#         # x = self.softmax(x)
#         # print(x)
#         x = self.dropout(bert_out)
#         x = self.fc1(x)

#         return x