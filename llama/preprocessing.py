# pip install underthesea

import re
import string
import pandas as pd
from nltk.corpus import stopwords

#Dowload stopwords
# import nltk
# nltk.download('stopwords')

def remove_punctuation(text):
    punctuation = string.punctuation
    return text.translate(str.maketrans('', '', punctuation))

with open('../vietnamese-stopwords/vietnamese-stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = f.read().split("\n")

stop_words = set(stopwords)

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

def remove_special_characters(text):
    # text = re.sub(r'\d+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s\u00C0-\u1EF9]', '', text)    
    return text

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = remove_punctuation(text)
    text = remove_special_characters(text)
    # text = remove_stopwords(text)
    return text

# data_path = '../data/UIT-VSFC/dev/dev_data.csv'

# data = pd.read_csv(
#     data_path,
#     encoding='utf-8',
#     encoding_errors='replace'
#     )

# print("Dữ liệu ban đầu khi chưa tiền xử lý:")
# print(data.head())
# print("Số lượng dữ liệu ban đầu: ",len(data))
# print("-------------------------------------------------------")

# data['text'] = data['text'].apply(preprocess_text)
# print("Dữ liệu sau khi tiền xử lý:")
# print(data.head(10))
# print("Số lượng dữ liệu sau khi tiền xử lý: ",len(data))

