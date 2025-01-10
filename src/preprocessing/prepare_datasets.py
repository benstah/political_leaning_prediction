import nltk
import os
import pandas as pd
import re
import string
# import numpy as np

from joblib import load, dump
from nltk.tokenize import word_tokenize   # nltk punkt must have been downloaded before
from nltk.corpus import stopwords         # nltk stopwords must have been downloaded before
from nltk.stem import PorterStemmer



# clean data
def _remove_amp(text):
    return text.replace("&amp;", " ")

def _remove_multiple_spaces(text):
    return re.sub(r'\s+', ' ', text)

def _remove_retweets(text):
    return re.sub(r'^RT[\s]+', ' ', text)

def _remove_links(text):
    return re.sub(r'https?:\/\/[^\s\n\r]+', ' ', text)

def _remove_hashes(text):
    return re.sub(r'#', ' ', text)

def _stitch_text_tokens_together(text_tokens):
    return " ".join(text_tokens)

def _tokenize(text):
    return nltk.word_tokenize(text, language="english")

def _stopword_filtering(text_tokens):
    stop_words = nltk.corpus.stopwords.words('english')

    return [token for token in text_tokens if token not in stop_words]

def _stemming(text_tokens):
    porter = nltk.stem.porter.PorterStemmer()
    return [porter.stem(token) for token in text_tokens]

def _remove_numbers(text):
    return re.sub(r'\d+', ' ', text)

def _lowercase(text):
    return text.lower()

def _remove_punctuation(text):
    return ''.join(character for character in text if character not in string.punctuation)


# preprocess all the data
def _preprocess(text):

    text = _remove_amp(text)
    text = _remove_links(text)
    text = _remove_hashes(text)
    text = _remove_multiple_spaces(text)

    text = _lowercase(text)
    text = _remove_punctuation(text)

    text_tokens = _tokenize(text)
    text_tokens = _stopword_filtering(text_tokens)
    text_tokens = _stemming(text_tokens)
    text = _stitch_text_tokens_together(text_tokens)

    return text.strip()



dirname = os.path.dirname(__file__)

# process data and safe it as new files
############ Training set small ############
df_training_s = load(dirname + '/../../data/interim/training_set_s')

# training_set = []
# for index, row in df_training_s.iterrows():
#     new_row = row.copy()
#     new_row.headline = _preprocess(str(row.headline))
#     new_row.lead = _preprocess(str(row.lead))
#     new_row.body = _preprocess(str(row.body))
#     training_set.append(new_row)
#     if (index % 1000) == 0:
#         print(index)

# df_training = pd.DataFrame(training_set)


print("step 1")
df_training_s["headline"] = df_training_s["headline"].astype(str)
df_training_s["lead"] = df_training_s["lead"].astype(str)
df_training_s["body"] = df_training_s["body"].astype(str)
# df_training_s["headline"] = [_preprocess(s) for s in df_training_s["headline"]]
# df_training_s["headline"] = np.vectorize(_preprocess)(df_training_s["headline"])

df_training_s["headline"] = df_training_s["headline"].map(lambda x: _preprocess(x))
print(df_training_s["headline"])
print("step 2")
df_training_s["lead"] = df_training_s["lead"].map(lambda x: _preprocess(x))
print("step 3")
df_training_s["body"] = df_training_s["body"].map(lambda x: _preprocess(x))


print("compress...")
dump(df_training_s, dirname + '/../../data/processed/training_set_s', compress=4) 
print("done...")

############ Training set large ############
df_training_l = load(dirname + '/../../data/interim/training_set_l')


df_training_l["headline"] = df_training_l["headline"].astype(str)
df_training_l["lead"] = df_training_l["lead"].astype(str)
df_training_l["body"] = df_training_l["body"].astype(str)
print("step 1")
df_training_l["headline"] = df_training_l["headline"].map(lambda x: _preprocess(x))
print("step 2")
df_training_l["lead"] = df_training_l["lead"].map(lambda x: _preprocess(x))
print("step 3")
df_training_l["body"] = df_training_l["body"].map(lambda x: _preprocess(x))

dump(df_training_l, dirname + '/../../data/processed/training_set_l', compress=4) 

############ Validation set ############
df_validation = load(dirname + '/../../data/interim/validation_set')

for index, row in df_validation.iterrows():
    df_validation.at[index, 'headline'] = _preprocess(str(row.headline))
    df_validation.at[index, 'lead'] = _preprocess(str(row.lead))
    df_validation.at[index, 'body'] = _preprocess(str(row.body))

dump(df_validation, dirname + '/../../data/processed/validation_set', compress=4) 


############ Test set ############
df_test = load(dirname + '/../../data/interim/test_set')

for index, row in df_test.iterrows():
    df_test.at[index, 'headline'] = _preprocess(str(row.headline))
    df_test.at[index, 'lead'] = _preprocess(str(row.lead))
    df_test.at[index, 'body'] = _preprocess(str(row.body))

dump(df_test, dirname + '/../../data/processed/test_set', compress=4) 





# print(df_training_s)
# print(df_training_s.iloc[0].headline)
# print(df_training_s.iloc[0].lead)
# print(df_training_s.iloc[100].headline)
# print(df_training_s.iloc[100].lead)
# exit()


# print(list(df_training_s))

# read_row = df_training_s.iloc[1000]
# print("headline" + read_row.headline)
# print("lead" + read_row.lead)
# print("###############################################")

# read_row = df_training_s.iloc[10000]
# print("headline" + read_row.headline)
# print("lead" + read_row.lead)
# print("###############################################")

# read_row = df_training_s.iloc[2000]
# print("headline" + read_row.headline)
# print("lead" + read_row.lead)
# print("###############################################")



# exit()

# # 1. lower case words
# sentence = sentence.lower()
# print(sentence)

# # 2. Tokenize words
# words = word_tokenize(sentence)
# print(words)

# # 3. filter out stop words
# filtered_sentence = [w for w in words if not w in stop_words] 
# print(filtered_sentence)

# # 3. stem words
# ps = PorterStemmer()
# stemmed = []
# for word in filtered_sentence:
#     stemmed.append(ps.stem(word))

# print(stemmed)