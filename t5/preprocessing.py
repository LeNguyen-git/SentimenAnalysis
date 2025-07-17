import pandas as pd
import re
import string
import unicodedata

class T5Preprocessing:
    def __init__(self):
        self.punctuation = string.punctuation
    
    def normalize_unicode(self, text):
        return unicodedata.normalize('NFC', text)

    def remove_special_characters(self, text):
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s\u00C0-\u1EF9]', '', text)
        return text

    def remove_punctuation(self, text):
        return text.translate(str.maketrans('', '', self.punctuation))
    
    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = self.normalize_unicode(text)
        text = self.remove_punctuation(text)
        text = self.remove_special_characters(text)     
        text = text.strip()

        return text

# data_path = "../data/UIT-VSFC/merge_data/all_data.csv"

# data = pd.read_csv(
#     data_path,
#     encoding='utf-8',
#     encoding_errors='replace'
# )

# processor = T5Preprocessing()

# print("Dữ liệu chưa được xử lý:")
# print(data.head())
# print("-" *50)
# data['text'] = data['text'].apply(processor.preprocess_test)
# print(f"Dữ liệu sau khi được xử lý:")
# print(data.head())

