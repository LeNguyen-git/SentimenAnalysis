import torch
import re
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import json
import os
from torch.optim import AdamW
import torch.nn as nn
from torch.utils.data import DataLoader



from t5_model import T5Model, ModelArgs
from t5_tokenizer import T5Tokenizer
from t5_preprocessing import T5Preprocessing
from t5_dataset import T5Dataset, create_dataloader
from t5_dataset_combine import T5Dataset_Combined, create_dataloader_combined

class T5Trainer:
    def __init__(self, model, tokenizer, train_dataloader, val_dataloader = None, device = None):
        
        self.model = model
        self.tokenizer = tokenizer

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        
        self.optimizer = AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-2) #1e-4
        self.loss_function = nn.CrossEntropyLoss(ignore_index=self.tokenizer.token_to_id["<pad>"])

        self.history = []

    def train_labels(self, epoch):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1} Training")

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            decoder_input_ids = labels[:, :-1].contiguous()
            decoder_labels = labels[:, 1:].contiguous()

            logits = self.model(input_ids, decoder_input_ids, attention_mask=attention_mask)

            loss = self.loss_function(logits.view(-1, logits.size(-1)), decoder_labels.view(-1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": total_loss / (progress_bar.n + 1)})

        avg_loss = total_loss / len(self.train_dataloader)
        return avg_loss
    
    def evaluate_labels(self):
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

                logits = self.model(input_ids, decoder_input_ids, attention_mask=attention_mask)
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
    
    def train_topics(self, epoch):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1} Training")

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            topics = batch['topics'].to(self.device)

            decoder_input_ids = topics[:, :-1].contiguous()
            decoder_topics = topics[:, 1:].contiguous()

            logits = self.model(input_ids, decoder_input_ids, attention_mask=attention_mask)

            loss = self.loss_function(logits.view(-1, logits.size(-1)), decoder_topics.view(-1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": total_loss / (progress_bar.n + 1)})

        avg_loss = total_loss / len(self.train_dataloader)
        return avg_loss
    
    def evaluate_topics(self):
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
                topics = batch['topics'].to(self.device)

                decoder_input_ids = topics[:, :-1].contiguous()
                decoder_topics = topics[:, 1:].contiguous()

                logits = self.model(input_ids, decoder_input_ids, attention_mask=attention_mask)
                loss = self.loss_function(logits.view(-1, logits.size(-1)), decoder_topics.view(-1))
                total_loss += loss.item()

                # Lấy dự đoán
                preds = torch.argmax(logits, dim=-1)
                mask = decoder_topics != self.tokenizer.token_to_id["<pad>"]
                predictions.extend(preds[mask].cpu().numpy())
                true_labels.extend(decoder_topics[mask].cpu().numpy())

        avg_loss = total_loss / len(self.val_dataloader)
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')

        return avg_loss, {"accuracy": accuracy, "f1": f1}
    
    def train_combined(self, epoch):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1} Training")

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            decoder_input_ids = labels[:, :-1].contiguous()
            decoder_labels = labels[:, 1:].contiguous()

            logits = self.model(input_ids, decoder_input_ids, attention_mask=attention_mask)
            loss = self.loss_function(logits.view(-1, logits.size(-1)), decoder_labels.view(-1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": total_loss / (progress_bar.n + 1)})

        avg_loss = total_loss / len(self.train_dataloader)
        return avg_loss

    def parse_label_text(self, label_text):
        pattern = r"Cảm xúc:\s*([^|]+)\|\s*Chủ đề:\s*(.+)"
        match = re.search(pattern, label_text)
        if match:
            sentiment = match.group(1).strip()
            topic = match.group(2).strip()
            return sentiment, topic
        else:
            parts = label_text.lower().split("|")
            sentiment = parts[0].replace("cảm xúc:", "").strip() if "cảm xúc:" in parts[0] else "unknown"
            topic = parts[1].replace("chủ đề:", "").strip() if len(parts) > 1 and "chủ đề:" in parts[1] else "unknown"
            return sentiment, topic


    def evaluate_combined(self):
        if self.val_dataloader is None:
            return None, None
        
        self.model.eval()
        total_loss = 0
        num_correct_tokens = 0
        num_total_tokens = 0
        all_predictions = []
        all_true_labels = []
        all_sentiment_preds = []
        all_sentiment_true = []
        all_topic_preds = []
        all_topic_true = []

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                decoder_input_ids = labels[:, :-1].contiguous()
                decoder_labels = labels[:, 1:].contiguous()

                logits = self.model(input_ids, decoder_input_ids, attention_mask=attention_mask)
                loss = self.loss_function(logits.view(-1, logits.size(-1)), decoder_labels.view(-1))
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                mask = decoder_labels != self.tokenizer.token_to_id["<pad>"]
                num_correct_tokens += ((preds == decoder_labels) & mask).sum().item()
                num_total_tokens += mask.sum().item()

                batch_size = input_ids.size(0)
                for i in range(batch_size):
                    pred_tokens = preds[i][mask[i]].cpu().numpy()
                    true_tokens = decoder_labels[i][mask[i]].cpu().numpy()
                    pred_text = self.tokenizer.decode(pred_tokens)
                    true_text = self.tokenizer.decode(true_tokens)

                    sentiment_pred, topic_pred = self.parse_label_text(pred_text)
                    sentiment_true, topic_true = self.parse_label_text(true_text)

                    all_predictions.append(f"{sentiment_pred}|{topic_pred}")
                    all_true_labels.append(f"{sentiment_true}|{topic_true}")
                    all_sentiment_preds.append(sentiment_pred)
                    all_sentiment_true.append(sentiment_true)
                    all_topic_preds.append(topic_pred)
                    all_topic_true.append(topic_true)

        avg_loss = total_loss / len(self.val_dataloader)
        token_accuracy = num_correct_tokens / num_total_tokens if num_total_tokens > 0 else 0
        overall_accuracy = accuracy_score(all_true_labels, all_predictions)
        sentiment_accuracy = accuracy_score(all_sentiment_true, all_sentiment_preds)
        sentiment_f1 = f1_score(all_sentiment_true, all_sentiment_preds, average='weighted', zero_division=0)
        topic_accuracy = accuracy_score(all_topic_true, all_topic_preds)
        topic_f1 = f1_score(all_topic_true, all_topic_preds, average='weighted', zero_division=0)

        return avg_loss, {
            'token_accuracy': token_accuracy,
            'overall_accuracy': overall_accuracy,
            'sentiment_accuracy': sentiment_accuracy,
            'sentiment_f1': sentiment_f1,
            'topic_accuracy': topic_accuracy,
            'topic_f1': topic_f1
        }


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

def main_labels():

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

    num_epochs = 3
    checkpoint_dir = 'checkpoints/t5_model_label.pth'
    history_path = 'training_history/t5_training_history_label.json'

    model = T5Model(model_args).to(device)
    trainer = T5Trainer(model, tokenizer, train_loader, val_loader, device=device)

    for epoch in range(num_epochs):
        train_loss = trainer.train_labels(epoch)
        val_loss, val_metrics = trainer.evaluate_labels()

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

def main_topics():

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
    checkpoint_dir  = 'checkpoints/t5_model_topics.pth'
    history_path  = 'training_history/t5_training_history_topics.json'

    model = T5Model(model_args).to(device)
    trainer = T5Trainer(model, tokenizer, train_loader, val_loader)

    for epoch in range(num_epochs):
        train_loss = trainer.train_topics(epoch)
        val_loss, val_metrics = trainer.evaluate_topics()

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

def main_combine():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = T5Tokenizer()
    tokenizer.load_vocab("../data/UIT-VSFC/t5_vocab.json")

    # Sử dụng T5Dataset_Combined
    train_dataset_combined = T5Dataset_Combined(
        data_path='../data/UIT-VSFC/merge_data/train_data.csv',
        tokenizer=tokenizer,
        max_length=256
    )

    train_loader_combined = create_dataloader_combined(
        train_dataset_combined,
        batch_size=8,
        shuffle=True
    )

    val_dataset_combined = T5Dataset_Combined(
        data_path='../data/UIT-VSFC/merge_data/dev_data.csv',
        tokenizer=tokenizer,
        max_length=256
    )

    val_loader_combined = create_dataloader_combined(
        val_dataset_combined,
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

    num_epochs = 3
    checkpoint_dir = 'checkpoints/t5_model_combined.pth'
    history_path = 'training_history/t5_training_history_combined.json'

    model = T5Model(model_args).to(device)
    trainer = T5Trainer(model, tokenizer, train_loader_combined, val_loader_combined, device=device)

    for epoch in range(num_epochs):
        train_loss = trainer.train_combined(epoch)
        val_loss, val_metrics = trainer.evaluate_combined()

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
    main_combine()


