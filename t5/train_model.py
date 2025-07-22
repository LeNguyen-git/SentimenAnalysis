import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import json
import os
from torch.optim import AdamW
import torch.nn as nn
from torch.utils.data import DataLoader



from model import T5Model, ModelArgs
from tokenizer import T5Tokenizer
from preprocessing import T5Preprocessing
from dataset import T5Dataset, create_dataloader

class T5Trainer:
    def __init__(self, model, tokenizer, train_dataloader, val_dataloader = None, device = None):
        
        self.model = model
        self.tokenizer = tokenizer

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.optimizer = AdamW(self.model.parameters(), lr=1e-4)
        self.loss_function = nn.CrossEntropyLoss(ignore_index=self.tokenizer.token_to_id["<pad>"])

        self.history = []

    def train(self, epoch):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1} Training")

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            decoder_input_ids = labels[:, :-1].contiguous()
            decoder_labels = labels[:, 1:].contiguous()

            logits = self.model(input_ids, decoder_input_ids)

            loss = self.loss_function(logits.view(-1, logits.size(-1)), decoder_labels.view(-1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": total_loss / (progress_bar.n + 1)})

        avg_loss = total_loss / len(self.train_dataloader)
        return avg_loss
    
    def evaluate(self):
        if self.val_dataloader is None:
            return None, None
        
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                decoder_input_ids = labels[:, :-1].contiguous()
                decoder_labels = labels[:, 1:].contiguous()

                logits = self.model(input_ids, decoder_input_ids)
                loss = self.loss_function(logits.view(-1, logits.size(-1)), decoder_labels.view(-1))
                total_loss += loss.item()

                # Lấy dự đoán
                preds = torch.argmax(logits, dim=-1)
                mask = decoder_labels != self.tokenizer.token_to_id["<pad>"]
                predictions.extend(preds[mask].cpu().numpy())
                true_labels.extend(decoder_labels[mask].cpu().numpy())

        avg_loss = total_loss / len(self.val_dataloader)
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')

        return avg_loss, {"accuracy": accuracy, "f1": f1}

    def save_checkpoint(self, epoch, loss, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }
        torch.save(checkpoint, path)

    def save_history(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=4)

    def load_checkpoint(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")
        else:
            print("Checkpoint file not found.")

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = T5Tokenizer()
    tokenizer.load_vocab("../data/UIT-VSFC/t5_vocab.json")

    train_dataset = T5Dataset(
        data_path='../data/UIT-VSFC/merge_data/train_data.csv',
        tokenizer=tokenizer,
        max_length=256
    )

    train_loader = create_dataloader( 
        train_dataset,
        batch_size=8,
        shuffle=True
    )

    val_dataset = T5Dataset(
        data_path='../data/UIT-VSFC/merge_data/dev_data.csv',
        tokenizer=tokenizer,
        max_length=256
    )

    val_loader = create_dataloader(
        val_dataset, 
        batch_size=8,
        shuffle=False
    )

    model_args = ModelArgs(
        vocab_size=len(tokenizer.token_to_id),
        d_model=256,
        num_heads=8,
        num_layers=6,
        hidden_dim=512,
        max_len=256,
        dropout=0.1,
        pad_idx=tokenizer.token_to_id["<pad>"]
    )

    num_epochs  = 3
    checkpoint_dir  = 'checkpoints/t5_model.pth'
    history_path  = 'training_history/t5_training_history.json'

    model = T5Model(model_args).to(device)
    trainer = T5Trainer(model, tokenizer, train_loader, val_loader)

    for epoch in range(num_epochs):
        train_loss = trainer.train(epoch)
        val_loss, val_metrics = trainer.evaluate()

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}")

        trainer.history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": val_metrics['accuracy'],
            "f1": val_metrics['f1']
        })

        trainer.save_checkpoint(epoch, val_loss, checkpoint_dir)

    trainer.save_history(history_path)



if __name__ == "__main__":
    main()





                

        