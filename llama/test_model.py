import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from model import MiniLlamaModel, ModelArgs
from dataset import LLamaDataset, create_dataloader
from tokenizer import LLaMaTokenizer

def test(model, test_loader, device, num_classes=3):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
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

    #Tính cho từng lớp
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)

    return (
        accuracy, precision, recall, f1, roc_auc, cm,
        precision_per_class, recall_per_class, f1_per_class
    )

def load_model(checkpoint_path, device, model_args):
    model = MiniLlamaModel(model_args).to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {checkpoint_path}")
    return model

def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = LLaMaTokenizer({})
    tokenizer.load_vocab('../data/UIT-VSFC/llama_vocab.json')

    test_dataset = LLamaDataset(
        data_path='../data/UIT-VSFC/merge_data/test_data.csv',
        tokenizer=tokenizer,
        max_length=128
    )
    test_loader = create_dataloader(test_dataset, batch_size=8, shuffle=False)

    model_args = ModelArgs(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=512,
        d_model=256,
        n_layers=6,
        n_heads=8,
        max_seq_len=128,
        num_classes=3,
        num_groups=4
    )

    checkpoint_path = 'checkpoints/model_4.pth'
    model = load_model(checkpoint_path, device, model_args)

    (
        accuracy, precision, recall, f1, roc_auc, cm,
        precision_per_class, recall_per_class, f1_per_class
    ) = test(model, test_loader, device, num_classes=3)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test ROC-AUC Score: {roc_auc:.4f}")

    print(f"\n📌 Kết quả từng lớp:")
    class_labels = ['Tiêu cực', 'Trung tính', 'Tích cực']
    for i, label in enumerate(class_labels):
        print(f"Lớp {label}:")
        print(f"  - Precision: {precision_per_class[i]:.4f}")
        print(f"  - Recall:    {recall_per_class[i]:.4f}")
        print(f"  - F1-score:  {f1_per_class[i]:.4f}")

    plot_confusion_matrix(cm, labels=['Tiêu cực (0)', 'Trung tính (1)', 'Tích cực (2)'])

if __name__ == "__main__":
    main()