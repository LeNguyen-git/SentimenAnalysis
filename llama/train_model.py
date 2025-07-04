#Import thư viên của pytorch
import os
import json
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


#Import thư viên cần thiết của mô hình MiniLLama
from model import MiniLlamaModel, ModelArgs
from dataset import LLamaDataset, create_dataloader
from tokenizer import LLaMaTokenizer
import preprocessing

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(train_loader, desc="Training")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask, labels)

        loss = outputs['loss']
        logits = outputs['logits']

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=-1).to(device)
        
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(train_loader)
    train_accuracy = accuracy_score(all_labels, all_preds)
    train_f1 = f1_score(all_labels, all_preds, average='weighted')

    return avg_loss, train_accuracy, train_f1

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, labels)

            loss = outputs['loss']
            logits = outputs['logits']

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1).to(device)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    val_loss = total_loss / len(val_loader)
    val_accuracy = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='weighted')

    return val_loss, val_accuracy, val_f1


def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved at {path}")

def load_checkpoint(model, optimizer, path):

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Checkpoint loaded from {path}, resuming from epoch {start_epoch}")

    return start_epoch

def training_history(train_loss, train_acc, train_f1, val_loss, val_acc, val_f1, epoch, history_file):
    history = {
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'train_accuracy': train_acc,
        'train_f1': train_f1,
        'val_loss': val_loss,
        'val_accuracy': val_acc,
        'val_f1': val_f1
    }
    
    if os.path.exists(history_file):
        with open(history_file, 'r', encoding='utf-8') as f:
            existing_history = json.load(f)
        existing_history.append(history)
    else:
        existing_history = [history]
    
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(existing_history, f, indent=4)
    print(f"Training history saved to {history_file}")

    


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = LLaMaTokenizer({})
    tokenizer.load_vocab('../data/UIT-VSFC/llama_vocab.json')

    train_dataset = LLamaDataset(
        data_path='../data/UIT-VSFC/merge_data/train_data.csv',
        tokenizer=tokenizer,
        max_length=128
    )

    val_dataset = LLamaDataset(
        data_path='../data/UIT-VSFC/merge_data/dev_data.csv',
        tokenizer=tokenizer,
        max_length=128
    )

    train_loader = create_dataloader(train_dataset, batch_size=8, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=8, shuffle=False)

    model_args = ModelArgs(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=256,
        d_model=256,
        n_layers=6,
        n_heads=8,
        max_seq_len=128,
        num_classes=3
    )

   
    num_epochs = 3
    checkpoint_path = 'checkpoints/model_2.pth'
    history_file = 'training_history/training_history_2.json'

    model = MiniLlamaModel(model_args).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    # scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss, train_accuracy, train_f1 = train(model, train_loader, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}")

        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}, Validation F1: {val_f1:.4f}")

        # scheduler.step()
        # print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        save_checkpoint(model, optimizer, epoch, checkpoint_path)

        training_history(train_loss, train_accuracy, train_f1, val_loss, val_acc, val_f1, epoch, history_file)
        

if __name__ == "__main__":
    main()