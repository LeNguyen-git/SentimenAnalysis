import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import json
import os



from bert_model import BertModel, ModelArgs
from bert_dataset import BertDataset, create_data_loader
from bert_tokenizer import BertTokenizer
import bert_preprocessing
    
def train_labels(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(train_loader, desc="Training")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        token_type_ids=token_type_ids, 
                        labels=labels,
                        )


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

def evaluate_labels(model, val_loader, device, output_attentions=False):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            token_type_ids=token_type_ids, 
                            labels=labels,
                            )

            loss = outputs['loss']
            logits = outputs['logits']

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1).to(device)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(val_loader)
    val_accuracy = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='weighted')

    return avg_loss, val_accuracy, val_f1

def train_topics(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_topics = []

    progress_bar = tqdm(train_loader, desc="Training")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        topics = batch['topics'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, topics=topics)


        loss = outputs['loss']
        logits = outputs['topic_logits']

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=-1).to(device)

        all_preds.extend(preds.cpu().tolist())
        all_topics.extend(topics.cpu().tolist())

        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(train_loader)
    train_accuracy = accuracy_score(all_topics, all_preds)
    train_f1 = f1_score(all_topics, all_preds, average='weighted')

    return avg_loss, train_accuracy, train_f1

def evaluate_topics(model, val_loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_topics = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            topics = batch['topics'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, topics=topics)

            loss = outputs['loss']
            logits = outputs['topic_logits']

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1).to(device)

            all_preds.extend(preds.cpu().tolist())
            all_topics.extend(topics.cpu().tolist())

    avg_loss = total_loss / len(val_loader)
    val_accuracy = accuracy_score(all_topics, all_preds)
    val_f1 = f1_score(all_topics, all_preds, average='weighted')

    return avg_loss, val_accuracy, val_f1

def save_checkpoint(model, optimizer, epoch, loss, accuracy, f1, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'accuracy': accuracy,
        'f1': f1
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    
    print(f"Checkpoint loaded from {path}")
    return start_epoch

def training_history(epoch,train_loss, train_accuracy, train_f1, val_loss, val_accuracy, val_f1, history_file):
    history = {
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'train_f1': train_f1,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
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


def main_labels():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = BertTokenizer(min_frequency=1)
    tokenizer.load_vocab('../data/UIT-VSFC/bert_vocab.json')

    train_data = BertDataset(
        data_path='../data/UIT-VSFC/merge_data/train_data.csv',
        tokenizer=tokenizer,
        max_length=256,
    )

    val_data = BertDataset(
        data_path='../data/UIT-VSFC/merge_data/dev_data.csv',
        tokenizer=tokenizer,
        max_length=256,
    )

    train_loader = create_data_loader(train_data, batch_size=8, shuffle=True)
    val_loader = create_data_loader(val_data, batch_size=8, shuffle=False)

    model_args = ModelArgs(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_layers=6,
        n_heads=8,
        hidden_dim=512,
        max_positions=512,
        type_vocab_size=2,
        dropout=0.1
    )

    num_epochs = 3
    checkpoint_path = 'checkpoints/bert_model_label.pth'
    history_file = 'training_history/bert_training_history_label.json'

    model = BertModel(model_args, num_labels=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2) #2e-5

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        train_loss, train_accuracy, train_f1 = train_labels(model, train_loader, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}")

        val_loss, val_accuracy, val_f1 = evaluate_labels(model, val_loader, device)
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")

        save_checkpoint(model, optimizer, epoch, train_loss, train_accuracy, train_f1, checkpoint_path)
        training_history(epoch, train_loss, train_accuracy, train_f1, val_loss, val_accuracy, val_f1, history_file)

def main_topics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = BertTokenizer(min_frequency=1)
    tokenizer.load_vocab('../data/UIT-VSFC/bert_vocab.json')

    train_data = BertDataset(
        data_path='../data/UIT-VSFC/merge_data/train_data.csv',  
        tokenizer=tokenizer,
        max_length=256,
    )

    val_data = BertDataset(
        data_path='../data/UIT-VSFC/merge_data/dev_data.csv',  
        tokenizer=tokenizer,
        max_length=256,
    )

    train_loader = create_data_loader(train_data, batch_size=8, shuffle=True)
    val_loader = create_data_loader(val_data, batch_size=8, shuffle=False)

    model_args = ModelArgs(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_layers=6,
        n_heads=8,
        hidden_dim=512,
        max_positions=512,
        type_vocab_size=2,
        dropout=0.1
    )

    num_epochs = 3
    checkpoint_path = 'checkpoints/bert_model_topic.pth'  
    history_file = 'training_history/bert_training_history_topic.json'  

    model = BertModel(model_args, num_topics=4).to(device)  
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        train_loss, train_accuracy, train_f1 = train_topics(model, train_loader, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}")

        val_loss, val_accuracy, val_f1 = evaluate_topics(model, val_loader, device)
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")

        save_checkpoint(model, optimizer, epoch, train_loss, train_accuracy, train_f1, checkpoint_path)
        training_history(epoch, train_loss, train_accuracy, train_f1, val_loss, val_accuracy, val_f1, history_file)

if __name__ == "__main__":
    main_labels()
    main_topics()
