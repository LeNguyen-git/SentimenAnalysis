import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import BertTokenizer
from preprocessing import TextPreprocessor

class BertDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        self.data = pd.read_csv(data_path, encoding='utf-8')
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocessor = TextPreprocessor()

        self.data['text'] = self.data['text'].apply(self.preprocessor.preprocess_text)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        texts = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['sentiment']

        token_ids = self.tokenizer.encode(texts, add_special_tokens=True)

        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids.extend([0] * (self.max_length - len(token_ids)))

        attention_mask = [1 if token_id != 0 else 0 for token_id in token_ids]
        # token_type_ids = [0] * self.max_length

        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            # 'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loader(dataset, batch_size=8, shuffle=True, num_workers=0):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
# if __name__ == "__main__":

#     tokenizer = BertTokenizer(min_frequency=1)
#     tokenizer.load_vocab('../data/UIT-VSFC/bert_vocab.json')

#     train_dataset = BertDataset(
#         data_path='../data/UIT-VSFC/merge_data/train_data.csv',
#         tokenizer=tokenizer,
#         max_length=128
#     )

#     train_loader = create_data_loader(train_dataset, batch_size=8, shuffle=True)

#     print("Train dataset size:", len(train_dataset))
#     print("Sample from train dataset:", train_dataset[0])

#     print("-"*50)

#     for batch in train_loader:
#         print("Input IDs:", batch['input_ids'][0])
#         print("Attention Mask:", batch['attention_mask'][0])
#         print("Token Type IDs:", batch['token_type_ids'][0])
#         print("Labels:", batch['labels'][0])
#         break
