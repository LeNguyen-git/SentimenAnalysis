import torch
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd
import json

from mapping_data import sentiment, topic
from preprocessing import T5Preprocessing
from tokenizer import T5Tokenizer

class T5Dataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=256):
        super().__init__()

        self.data = pd.read_csv(data_path, encoding='utf-8', encoding_errors='replace')
        self.max_length = max_length
        self.preprocessor = T5Preprocessing()
        self.tokenizer = tokenizer

        self.texts = self.data['text'].apply(self.preprocessor.preprocess_text).tolist()
        self.labels = self.data['sentiment'].apply(sentiment).tolist()

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        input_text = f"Phân tích cảm xúc cho đoạn văn bản sau: {text}"

        input_ids = self.tokenizer.encode(
            input_text,
            add_special_tokens = True
        )

        input_ids = input_ids[:self.max_length]
        input_ids += [self.tokenizer.token_to_id["<pad>"]] * (self.max_length - len(input_ids))

        attention_mask = [1 if id != self.tokenizer.token_to_id["<pad>"] else 0 for id in input_ids]


        label_ids = self.tokenizer.encode(
            label,
            add_special_tokens = True
        )

        label_ids = label_ids[:self.max_length]
        label_ids += [self.tokenizer.token_to_id["<pad>"]] * (self.max_length - len(label_ids))

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        label_ids = torch.tensor(label_ids, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label_ids
        }
    
def create_dataloader(dataset, batch_size=8, shuffle=True, num_workers=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )



# if __name__ == "__main__":

#     tokenizer = T5Tokenizer()
#     tokenizer.load_vocab("../data/UIT-VSFC/t5_vocab.json")

#     train_path = "../data/UIT-VSFC/merge_data/train_data.csv"

#     train_dataset = create_dataloader(train_path, tokenizer, batch_size=8, max_length=256)
#     print("Số lượng batch:", len(train_dataset))
#     for batch in train_dataset:
#         print("input_ids:", batch['input_ids'])
#         print("attention_mask:", batch['attention_mask'])
#         print("labels:", batch['labels'])
#         break 







        

    