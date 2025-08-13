import torch
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd
import json

from t5_mapping_data import sentiment, topic
from t5_preprocessing import T5Preprocessing
from t5_tokenizer import T5Tokenizer

class T5Dataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=256):
        super().__init__()

        self.data = pd.read_csv(data_path, encoding='utf-8', encoding_errors='replace')
        self.max_length = max_length
        self.preprocessor = T5Preprocessing()
        self.tokenizer = tokenizer

        self.texts = self.data['text'].apply(self.preprocessor.preprocess_text).tolist()
        self.labels = self.data['sentiment'].apply(sentiment).tolist()
        self.topics = self.data['topic'].apply(topic).tolist()

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        topic = self.topics[index]

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

        topic_ids = self.tokenizer.encode(
            topic,
            add_special_tokens = True
        )

        topic_ids = topic_ids[:self.max_length]
        topic_ids += [self.tokenizer.token_to_id["<pad>"]] * (self.max_length - len(topic_ids))




        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        label_ids = torch.tensor(label_ids, dtype=torch.long)
        topic_ids = torch.tensor(topic_ids, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label_ids,
            'topics': topic_ids
        }
    
def create_dataloader(dataset, batch_size=8, shuffle=True, num_workers=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

def main():

    tokenizer = T5Tokenizer()
    tokenizer.load_vocab("../data/UIT-VSFC/t5_vocab.json")

    train_path = "../data/UIT-VSFC/merge_data/train_data.csv"

    train_dataset = T5Dataset(
        data_path=train_path,
        tokenizer=tokenizer,
        max_length=256
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=8)

    print("Số lượng batch:", len(train_loader))
    for batch in train_loader:
        print("input_ids:", batch['input_ids'])
        print("attention_mask:", batch['attention_mask'])
        print("labels:", batch['labels'])
        print("topics:", batch['topics'])
        break 


if __name__ == "__main__":
    main()







        

    