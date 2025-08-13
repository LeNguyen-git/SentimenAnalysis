import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import random
from bert_tokenizer import BertTokenizer
from bert_preprocessing import TextPreprocessor

class BertDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128, mlm_probability=0.15, with_mlm=False):
        self.data = pd.read_csv(data_path, encoding='utf-8')
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.with_mlm = with_mlm
        self.preprocessor = TextPreprocessor()

        self.data['text'] = self.data['text'].apply(self.preprocessor.preprocess_text)
        
        self.mask_token_id = tokenizer.token_to_id("[MASK]")  # ID của token [MASK]
        self.vocab_size = tokenizer.vocab_size

    def __len__(self):
        return len(self.data)
    
    def mask_tokens(self, token_ids):
        labels = token_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        special_tokens_mask = (token_ids == 0) | (token_ids == 1) | (token_ids == 2)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Chỉ loss ở vị trí bị mask

        # 80% -> [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        token_ids[indices_replaced] = self.mask_token_id

        # 10% -> random token
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.vocab_size, labels.shape, dtype=torch.long)
        token_ids[indices_random] = random_words[indices_random]

        return token_ids, labels

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['sentiment']
        topic = self.data.iloc[idx]['topic']

        token_ids = self.tokenizer.encode(text, add_special_tokens=True)

        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids.extend([0] * (self.max_length - len(token_ids)))

        attention_mask = [1 if tid != 0 else 0 for tid in token_ids]
        token_type_ids = [0] * self.max_length

        token_ids_tensor = torch.tensor(token_ids, dtype=torch.long)

        if self.with_mlm:
            input_ids_mlm, mlm_labels = self.mask_tokens(token_ids_tensor.clone())
            return {
                'input_ids': input_ids_mlm,
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'labels': torch.tensor(label, dtype=torch.long),
                'topics': torch.tensor(topic, dtype=torch.long),
                'mlm_labels': mlm_labels
            }
        else:
            return {
                'input_ids': token_ids_tensor,
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'labels': torch.tensor(label, dtype=torch.long),
                'topics': torch.tensor(topic, dtype=torch.long)
            }

def create_data_loader(dataset, batch_size=8, shuffle=True, num_workers=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )



def main():
        tokenizer = BertTokenizer(min_frequency=1)
        tokenizer.load_vocab('../data/UIT-VSFC/bert_vocab.json')

        train_dataset = BertDataset(
            data_path='../data/UIT-VSFC/merge_data/train_data.csv',
            tokenizer=tokenizer,
            max_length=128
        )

        train_loader = create_data_loader(train_dataset, batch_size=8, shuffle=True)

        print("Train dataset size:", len(train_dataset))
        print("Sample from train dataset:", train_dataset[0])

        print("-"*50)

        for batch in train_loader:
            print("Input IDs:", batch['input_ids'][0])
            print("Attention Mask:", batch['attention_mask'][0])
            print("Token Type IDs:", batch['token_type_ids'][0])
            print("Labels:", batch['labels'][0])
            print("Topics:", batch['topics'][0])
            break

if __name__ == "__main__":
    main()

