import torch
import numpy as np 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torch.nn as nn

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from model import T5Model, ModelArgs
from dataset import T5Dataset, create_dataloader
from tokenizer import T5Tokenizer
from preprocessing import T5Preprocessing
from mapping_data import sentiment, topic, reverse_sentiment, reverse_topic

class T5Tester:
    def __init__(self, model, tokenizer, test_dataloader, device):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.test_dataloader = test_dataloader
        self.loss_function = nn.CrossEntropyLoss(ignore_index=self.tokenizer.token_to_id["<pad>"])

    def test(self):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_true_labels = []
        predicted_texts = []
        true_texts = []

        with torch.no_grad():
            progress_bar = tqdm(self.test_dataloader, desc="Testing")
            
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                decoder_input_ids = labels[:, :-1].contiguous()
                decoder_labels = labels[:, 1:].contiguous()

                # Forward pass
                logits = self.model(input_ids, decoder_input_ids)
                
                # Tính loss
                loss = self.loss_function(logits.view(-1, logits.size(-1)), decoder_labels.view(-1))
                total_loss += loss.item()

                # Lấy predictions
                predictions = torch.argmax(logits, dim=-1)
                
                for i in range(predictions.shape[0]):

                    pred_ids = predictions[i].cpu().tolist()
                    pred_text = self.tokenizer.decode(pred_ids).strip()
                    predicted_texts.append(pred_text)
                    
                    true_ids = labels[i].cpu().tolist()
                    true_text = self.tokenizer.decode(true_ids).strip()
                    true_texts.append(true_text)
                    
                    try:
                        pred_sentiment = reverse_sentiment(pred_text)
                        true_sentiment = reverse_sentiment(true_text)
                    except:
                        pred_sentiment = 1 
                        true_sentiment = 1 
                    
                    all_predictions.append(pred_sentiment)
                    all_true_labels.append(true_sentiment)

                progress_bar.set_postfix({"loss": total_loss / (progress_bar.n + 1)})

        # Tính metrics
        avg_loss = total_loss / len(self.test_dataloader)
        accuracy = accuracy_score(all_true_labels, all_predictions)
        precision = precision_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
        
        cm = confusion_matrix(all_true_labels, all_predictions)
        
        print(f"\n{'='*50}")
        print("TEST RESULTS")
        print(f"{'='*50}")
        print(f"Test Loss: {avg_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
                
        labels = ['Tiêu cực', 'Trung tính', 'Tích cực']
        self.plot_confusion_matrix(cm, labels)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'predictions': predicted_texts,
            'true_labels': true_texts
        }

    def generate_text(self, input_text, max_length=50):
        self.model.eval()
        
        preprocessor = T5Preprocessing()
        processed_text = preprocessor.preprocess_text(input_text)
        input_prompt = f"Phân tích cảm xúc cho đoạn văn bản sau: {processed_text}"
        
        input_ids = self.tokenizer.encode(input_prompt, add_special_tokens=True)
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        decoder_input_ids = torch.tensor([[self.tokenizer.token_to_id["<s>"]]], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            for _ in range(max_length):
                logits = self.model(input_ids, decoder_input_ids)
                
                next_token_logits = logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                
                if next_token_id.item() == self.tokenizer.token_to_id["</s>"]:
                    break
                
                decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=-1)
        
        generated_ids = decoder_input_ids[0].cpu().tolist()
        generated_text = self.tokenizer.decode(generated_ids)
        
        return generated_text.strip()

 
    @staticmethod
    def load_model(checkpoint_path, device, model_args):
        model = T5Model(model_args).to(device)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {checkpoint_path}")
        return model
    
    @staticmethod
    def plot_confusion_matrix(cm, labels):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()
    

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = T5Tokenizer()
    tokenizer.load_vocab('../data/UIT-VSFC/t5_vocab.json')

    test_dataset = T5Dataset(
        data_path='../data/UIT-VSFC/merge_data/test_data.csv',
        tokenizer=tokenizer,
        max_length=256
    )

    test_loader = create_dataloader(
        test_dataset,
        batch_size= 8,
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

    checkpoint_path = 'checkpoints/t5_model.pth'

    model = T5Tester.load_model(checkpoint_path, device, model_args)

    tester = T5Tester(model, tokenizer, test_loader, device)
    tester.test()

if __name__ == "__main__":
    main()
        
