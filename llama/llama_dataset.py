from torch.utils.data import Dataset, DataLoader
import json
import os
import torch
import pandas as pd
import numpy as np
from llama_tokenizer import LLaMaTokenizer
import llama_preprocessing


class LLamaDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        self.data = pd.read_csv(data_path, encoding='utf-8')
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.data['text'] = self.data['text'].apply(llama_preprocessing.preprocess_text)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
            texts = self.data.iloc[idx]['text']
            sentiment = self.data.iloc[idx]['sentiment']
            topic = self.data.iloc[idx]['topic']

            token_ids = self.tokenizer.encode(texts, add_special_tokens=True)

            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length-1] + [self.tokenizer.str_to_int['</s>']]
            else:
                pad_length = self.max_length - len(token_ids)
                token_ids = token_ids + [self.tokenizer.str_to_int['<pad>']] * pad_length

            attention_mask = [1 if token_id != self.tokenizer.str_to_int['<pad>'] else 0 
                         for token_id in token_ids]
            
            return {
                'input_ids': torch.tensor(token_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'labels': torch.tensor(sentiment, dtype=torch.long),
                'topics': torch.tensor(topic, dtype=torch.long)
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
    
    tokenizer = LLaMaTokenizer({})
    tokenizer.load_vocab('../data/UIT-VSFC/llama_vocab.json')

    train_dataset = LLamaDataset(
        data_path='../data/UIT-VSFC/merge_data/train_data.csv',
        tokenizer=tokenizer,
        max_length=128
    )

    dev_dataset = LLamaDataset(
        data_path='../data/UIT-VSFC/merge_data/dev_data.csv',
        tokenizer=tokenizer,
        max_length=128
    )

    test_dataset = LLamaDataset(
        data_path='../data/UIT-VSFC/merge_data/test_data.csv',
        tokenizer=tokenizer,
        max_length=128
    )

    train_loader = create_dataloader(train_dataset, batch_size=8, shuffle=True)

    dev_loader = create_dataloader(dev_dataset, batch_size=8, shuffle=False)

    test_loader = create_dataloader(test_dataset, batch_size=8, shuffle=False)

    print("Train dataset size:", len(train_dataset))
    print("Sample from train dataset:", train_dataset[0])
    print("Dev dataset size:", len(dev_dataset))
    print("Sample from dev dataset:", dev_dataset[0])
    print("Test dataset size:", len(test_dataset))
    print("Sample from test dataset:", test_dataset[0])

    print(f"DataLoader created with {len(train_loader)} batches")

    print("-"*50)

    for batch in train_loader:
        print("Input IDs:", batch['input_ids'][0])         
        print("Attention Mask:", batch['attention_mask'][0])
        print("Labels:", batch['labels'][0])
        print("Topics:", batch['topics'][0])
        break 
    
if __name__ == "__main__":
     main()


    
    
           

