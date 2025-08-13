import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from bert_model import BertModel, ModelArgs
from bert_dataset import BertDataset, create_data_loader
from bert_tokenizer import BertTokenizer


def test_labels(model, test_loader, device, num_classes=3):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)                        
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=-1)

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().numpy())
        
    all_probs = np.array(all_probs)
    all_labels_bin = label_binarize(all_labels, classes=range(num_classes))

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    roc_auc = roc_auc_score(all_labels_bin, all_probs, average='weighted', multi_class='ovr')

    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))

    # Theo từng nhãn
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)

    return accuracy, precision, recall, f1, roc_auc, cm, precision_per_class, recall_per_class, f1_per_class

def test_topics(model, test_loader, device, num_topics=4):
    model.eval()
    all_preds = []
    all_topics = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)                        
            token_type_ids = batch['token_type_ids'].to(device)
            topics = batch['topics'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, topics=topics)
            logits = outputs['topic_logits']
            probs = torch.softmax(logits, dim=-1)

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_topics.extend(topics.cpu().tolist())
            all_probs.extend(probs.cpu().numpy())
        
    all_probs = np.array(all_probs)
    all_topics_bin = label_binarize(all_topics, classes=range(num_topics))

    accuracy = accuracy_score(all_topics, all_preds)
    precision = precision_score(all_topics, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_topics, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_topics, all_preds, average='weighted', zero_division=0)
    roc_auc = roc_auc_score(all_topics_bin, all_probs, average='weighted', multi_class='ovr')

    cm = confusion_matrix(all_topics, all_preds, labels=range(num_topics))

    precision_per_class = precision_score(all_topics, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_topics, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_topics, all_preds, average=None, zero_division=0)

    return accuracy, precision, recall, f1, roc_auc, cm, precision_per_class, recall_per_class, f1_per_class


def load_model(checkpoint_path, device, model_args, num_labels=None, num_topics=None):
    model = BertModel(model_args, num_labels=num_labels, num_topics=num_topics).to(device) 
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {checkpoint_path}")
    return model

def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90, va='center')

    plt.tight_layout()
    plt.show()


def main_label():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = BertTokenizer(min_frequency=1)
    tokenizer.load_vocab('../data/UIT-VSFC/bert_vocab.json')

    test_dataset = BertDataset(
        data_path='../data/UIT-VSFC/merge_data/test_data.csv',
        tokenizer=tokenizer,
        max_length=256,
    )

    test_loader = create_data_loader(test_dataset, batch_size=8, shuffle=False)

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

    checkpoint_path = 'checkpoints/bert_model_label.pth'
    model = load_model(checkpoint_path, device, model_args, num_labels=3)

    accuracy, precision, recall, f1, roc_auc, cm, precision_per_class, recall_per_class, f1_per_class = test_labels(model, test_loader, device, num_classes=3)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    print(f"\n📌 Kết quả theo từng nhãn:")
    class_names = ['Tiêu cực', 'Trung tính', 'Tích cực']
    for i, label in enumerate(class_names):
        print(f"Lớp {label}:")
        print(f"  - Precision: {precision_per_class[i]:.4f}")
        print(f"  - Recall:    {recall_per_class[i]:.4f}")
        print(f"  - F1-score:  {f1_per_class[i]:.4f}")

    plot_confusion_matrix(cm, labels=['Tiêu cực (0)', 'Trung tính(1)', 'Tích cực (2)'])


def main_topics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = BertTokenizer(min_frequency=1)
    tokenizer.load_vocab('../data/UIT-VSFC/bert_vocab.json')

    test_dataset = BertDataset(
        data_path='../data/UIT-VSFC/merge_data/test_data.csv', 
        tokenizer=tokenizer,
        max_length=256,
    )

    test_loader = create_data_loader(test_dataset, batch_size=8, shuffle=False)

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

    checkpoint_path = 'checkpoints/bert_model_topic.pth'  
    model = load_model(checkpoint_path, device, model_args, num_topics=4)  

    accuracy, precision, recall, f1, roc_auc, cm, precision_per_class, recall_per_class, f1_per_class = test_topics(model, test_loader, device, num_topics=4)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    print(f"\n📌 Kết quả theo từng nhãn:")
    class_names = ['Giảng viên', 'Chương trình học', 'Cơ sở vật chất', 'Khác']  
    for i, label in enumerate(class_names):
        print(f"Lớp {label}:")
        print(f"  - Precision: {precision_per_class[i]:.4f}")
        print(f"  - Recall:    {recall_per_class[i]:.4f}")
        print(f"  - F1-score:  {f1_per_class[i]:.4f}")

    plot_confusion_matrix(cm, labels=['Giảng viên (0)', 'Chương trình học (1)', 'Cơ sở vật chất (2)', 'Khác (3)'])

if __name__ == "__main__":
    #main_label()
    main_topics()