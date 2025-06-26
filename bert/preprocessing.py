import pandas as pd
import re
import string
import unicodedata

class TextPreprocessor:
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
        text = self.remove_punctuation(text)
        text = self.remove_special_characters(text)
        return text


# data_path = '../data/UIT-VSFC/merge_data/all_data.csv'

# data = pd.read_csv(
#     data_path,
#     encoding= 'utf-8',
#     encoding_errors= 'replace'
# )

# processor = TextPreprocessor()


# print("Dữ liệu chưa được sử lý:")
# print(data.head())
# print("-" * 30)
# data['text'] = data['text'].apply(processor.preprocess_text)
# print("Dữ liệu sau khi tiền xử lý:")
# print(data.head())