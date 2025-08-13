import torch
import re
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
import torch.nn as nn

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from t5_model import T5Model, ModelArgs
from t5_dataset import T5Dataset, create_dataloader
from t5_dataset_combine import T5Dataset_Combined, create_dataloader_combined
from t5_tokenizer import T5Tokenizer
from t5_preprocessing import T5Preprocessing
from t5_mapping_data import sentiment, topic, reverse_sentiment, reverse_topic, reverse_sentiment_fuzzy, reverse_topic_fuzzy


class T5Tester:
    def __init__(self, model, tokenizer, test_dataloader, device):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.test_dataloader = test_dataloader
        self.loss_function = nn.CrossEntropyLoss(ignore_index=self.tokenizer.token_to_id["<pad>"])

    def test_for_labels(self):
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

                logits = self.model(input_ids, decoder_input_ids, attention_mask=attention_mask)
                loss = self.loss_function(logits.view(-1, logits.size(-1)), decoder_labels.view(-1))
                total_loss += loss.item()

                predictions = torch.argmax(logits, dim=-1)

                for i in range(predictions.shape[0]):

                    pred_ids = predictions[i].cpu().tolist()
                    pred_text_raw = self.tokenizer.decode(pred_ids)

                    pred_text_clean = pred_text_raw.split("</s>")[0]
                    pred_text_clean = pred_text_clean.replace("<s>", "").replace("<pad>", "").strip()
                    pred_label = pred_text_clean

                    predicted_texts.append(pred_label)

                    true_ids = labels[i].cpu().tolist()
                    true_text_raw = self.tokenizer.decode(true_ids)

                    true_text_clean = true_text_raw.split("</s>")[0]
                    true_text_clean = true_text_clean.replace("<s>", "").replace("<pad>", "").strip()
                    true_label = true_text_clean

                    true_texts.append(true_label)


                    try:
                        pred_sentiment = reverse_sentiment(pred_label)
                        true_sentiment = reverse_sentiment(true_label)
                                            
                        all_predictions.append(pred_sentiment)
                        all_true_labels.append(true_sentiment)
                    except:
                        print(f"Skipping sample: pred='{pred_label}', true='{true_label}'")
                        continue


                    
                progress_bar.set_postfix({"loss": total_loss / (progress_bar.n + 1)})

        print("\n[DEBUG] Unique predicted label strings:")
        print(f"Số lượng: {len(set(predicted_texts))}")
        print(set(predicted_texts))

        print("\n[DEBUG] Unique true label strings:")
        print(f"Số lượng: {len(set(true_texts))}")
        print(set(true_texts))


        avg_loss = total_loss / len(self.test_dataloader)
        accuracy = accuracy_score(all_true_labels, all_predictions)
        precision = precision_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_true_labels, all_predictions, average='weighted', zero_division=0)

        # precision_per_class = precision_score(all_true_labels, all_predictions, average=None, zero_division=0)
        # recall_per_class = recall_score(all_true_labels, all_predictions, average=None, zero_division=0)
        # f1_per_class = f1_score(all_true_labels, all_predictions, average=None, zero_division=0)

        labels_index = [0, 1, 2]
        labels_str = ['Tiêu cực', 'Trung tính', 'Tích cực']

        cm = confusion_matrix(all_true_labels, all_predictions, labels=labels_index)

        print(f"\n{'='*50}")
        print("TEST RESULTS")
        print(f"{'='*50}")
        print(f"Test Loss: {avg_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")


        print("\nClassification Report:")
        print(classification_report(all_true_labels, all_predictions, target_names=labels_str))

        self.plot_confusion_matrix(cm, labels_str)

        return {
            # sentiment results
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            # 'precision_per_class': precision_per_class,
            # 'recall_per_class': recall_per_class,
            # 'f1_per_class': f1_per_class,
            'predictions': predicted_texts,
            'true_labels': true_texts,
        }

    def test_for_topics(self):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_true_topics = []
        predicted_texts = []
        true_texts = []
        pred_topic_str_list = []
        true_topic_str_list = []

        with torch.no_grad():
            progress_bar = tqdm(self.test_dataloader, desc="Testing")

            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                topics = batch['topics'].to(self.device)

                decoder_input_ids = topics[:, :-1].contiguous()
                decoder_topics = topics[:, 1:].contiguous()

                logits = self.model(input_ids, decoder_input_ids, attention_mask=attention_mask)
                loss = self.loss_function(logits.view(-1, logits.size(-1)), decoder_topics.view(-1))
                total_loss += loss.item()

                predictions = torch.argmax(logits, dim=-1)

                for i in range(predictions.shape[0]):
                    pred_ids = predictions[i].cpu().tolist()

                    pred_text_raw = self.tokenizer.decode(pred_ids)
                    pred_topic = pred_text_raw.split("</s>")[0].replace("<s>", "").replace("<pad>", "").strip()
                    predicted_texts.append(pred_topic) 

                    true_ids = topics[i].cpu().tolist()

                    true_text_raw = self.tokenizer.decode(true_ids)
                    true_topic = true_text_raw.split("</s>")[0].replace("<s>", "").replace("<pad>", "").strip()
                    true_texts.append(true_topic)


                    try:
                        pred_topic_idx, pred_topic_str = reverse_topic_fuzzy(pred_topic)
                        true_topic_idx, true_topic_str = reverse_topic_fuzzy(true_topic)

                        pred_topic_str_list.append(pred_topic_str)
                        true_topic_str_list.append(true_topic_str)

                        if pred_topic_idx == -1 or true_topic_idx == -1:
                            print(f"Skipping sample: pred='{pred_topic}', true='{true_topic}'")
                            continue

                        all_predictions.append(pred_topic_idx)
                        all_true_topics.append(true_topic_idx)
                    except Exception as e:
                        print(f"Skipping sample: pred='{pred_topic}', true='{true_topic}', error={str(e)}")
                        continue

                progress_bar.set_postfix({"loss": total_loss / (progress_bar.n + 1)})


        print("\n[DEBUG] Unique predicted label strings:")
        print(f"Số lượng: {len(set(predicted_texts))}")
        print(set(predicted_texts))

        print("\n[DEBUG] Unique true label strings:")
        print(f"Số lượng: {len(set(true_texts))}")
        print(set(true_texts))

        print("\n[DEBUG] Unique pred_topic_str values:")
        print(f"Số lượng: {len(set(pred_topic_str_list))}")
        print(set(pred_topic_str_list))

        print("\n[DEBUG] Unique true_topic_str values:")
        print(f"Số lượng: {len(set(true_topic_str_list))}")
        print(set(true_topic_str_list))


        topics_index = [0, 1, 2, 3]
        topics_str = ['Giảng viên', 'Chương trình học', 'Cơ sở vật chất', 'Khác']

        avg_loss = total_loss / len(self.test_dataloader)
        accuracy = accuracy_score(all_true_topics, all_predictions)
        precision = precision_score(all_true_topics, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_true_topics, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_true_topics, all_predictions, average='weighted', zero_division=0)

        precision_per_class = precision_score(all_true_topics, all_predictions, average=None, labels=topics_index, zero_division=0)
        recall_per_class = recall_score(all_true_topics, all_predictions, average=None, labels=topics_index, zero_division=0)
        f1_per_class = f1_score(all_true_topics, all_predictions, average=None, labels=topics_index, zero_division=0)

        cm = confusion_matrix(all_true_topics, all_predictions, labels=topics_index)

        print(f"\n{'='*50}")
        print("TEST RESULTS")
        print(f"{'='*50}")
        print(f"Test Loss: {avg_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        print(f"\nKết quả từng lớp:")
        for i, label_idx in enumerate(topics_index):
            print(f"Lớp {topics_str[i]}:")
            print(f"  - Precision: {precision_per_class[i]:.4f}")
            print(f"  - Recall:    {recall_per_class[i]:.4f}")
            print(f"  - F1-score:  {f1_per_class[i]:.4f}")

        print("\nClassification Report:")
        print(classification_report(all_true_topics, all_predictions, target_names=topics_str))

        self.plot_confusion_matrix(cm, topics_str)

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'predictions': predicted_texts,
            'true_labels': true_texts,
        }


    # def generate_text(self, input_text, max_length=50):
    #     self.model.eval()

    #     preprocessor = T5Preprocessing()
    #     processed_text = preprocessor.preprocess_text(input_text)
    #     input_prompt = f"Phân tích cảm xúc cho đoạn văn bản sau: {processed_text}"

    #     input_ids = self.tokenizer.encode(input_prompt, add_special_tokens=True)
    #     input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)


    #     decoder_input_ids = torch.tensor([[self.tokenizer.token_to_id["<s>"]]], dtype=torch.long).to(self.device)

    #     generated_ids = []

    #     consecutive_eos_count = 0 

    #     with torch.no_grad():
    #         for _ in range(max_length):
    #             logits = self.model(input_ids, decoder_input_ids)
    #             next_token_logits = logits[:, -1, :]
    #             next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

    #             generated_ids.append(next_token_id.item())

    #             # Đếm liên tiếp </s>
    #             if next_token_id.item() == self.tokenizer.token_to_id["</s>"]:
    #                 consecutive_eos_count += 1
    #             else:
    #                 consecutive_eos_count = 0

    #             if consecutive_eos_count >= 3:
    #                 break

    #             decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=-1)


    #     generated_text = self.tokenizer.decode(generated_ids)
    #     return generated_text.strip()

    def parse_label_text(self, label_text):
        pattern = r"Cảm xúc:\s*([^|]+)\|\s*Chủ đề:\s*(.+)"
        match = re.search(pattern, label_text)

        if match:
            sentiment = match.group(1).strip()
            topic = match.group(2).strip()
            return sentiment, topic
        else:
            # if "Cảm xúc:" in label_text and "Chủ đề:" in label_text:
            #     parts = label_text.split("Chủ đề:")
            #     sentiment = parts[0].replace("Cảm xúc:", "").strip().replace("|", "").strip()
            #     topic = parts[1].strip() if len(parts) > 1 else "unknown"
            #     return sentiment, topic

            parts = label_text.lower().split("|")
            sentiment = parts[0].replace("cảm xúc:", "").strip() if "cảm xúc:" in parts[0] else "unknown"
            topic = parts[1].replace("chủ đề:", "").strip() if len(parts) > 1 and "chủ đề:" in parts[1] else "unknown"
            return sentiment, topic
            
            # return "unknown", "unknown"

    def test_combined(self, test_dataloader):
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_true_labels = []
        all_sentiment_preds = []
        all_sentiment_true = []
        all_topic_preds = []
        all_topic_true = []
        
        num_correct_tokens = 0
        num_total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Evaluating"):
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
        
        # --- Gộp tính toán metrics ---
        avg_loss = total_loss / len(test_dataloader)
        token_accuracy = num_correct_tokens / num_total_tokens if num_total_tokens > 0 else 0
        overall_accuracy = accuracy_score(all_true_labels, all_predictions)

        sentiment_accuracy = accuracy_score(all_sentiment_true, all_sentiment_preds)
        sentiment_f1 = f1_score(all_sentiment_true, all_sentiment_preds, average='weighted', zero_division=0)
        sentiment_precision = precision_score(all_sentiment_true, all_sentiment_preds, average='weighted', zero_division=0)
        sentiment_recall = recall_score(all_sentiment_true, all_sentiment_preds, average='weighted', zero_division=0)

        topic_accuracy = accuracy_score(all_topic_true, all_topic_preds)
        topic_f1 = f1_score(all_topic_true, all_topic_preds, average='weighted', zero_division=0)
        topic_precision = precision_score(all_topic_true, all_topic_preds, average='weighted', zero_division=0)
        topic_recall = recall_score(all_topic_true, all_topic_preds, average='weighted', zero_division=0)

        metrics = {
            'loss': avg_loss,
            'token_accuracy': token_accuracy,
            'overall_accuracy': overall_accuracy,
            'sentiment': {
                'accuracy': sentiment_accuracy,
                'f1': sentiment_f1,
                'precision': sentiment_precision,
                'recall': sentiment_recall
            },
            'topic': {
                'accuracy': topic_accuracy,
                'f1': topic_f1,
                'precision': topic_precision,
                'recall': topic_recall
            }
        }
        
        # --- Gộp in kết quả ---
        print("\n" + "="*80)
        print("MODEL EVALUATION RESULTS")
        print("="*80)
        print(f"Average Loss: {metrics['loss']:.4f}")
        print(f"Token-level Accuracy: {metrics['token_accuracy']:.4f}")
        print(f"Overall Exact Match Accuracy: {metrics['overall_accuracy']:.4f}")
        
        print(f"\nSentiment Analysis Performance:")
        print(f"  Accuracy:  {metrics['sentiment']['accuracy']:.4f}")
        print(f"  F1-Score:  {metrics['sentiment']['f1']:.4f}")
        print(f"  Precision: {metrics['sentiment']['precision']:.4f}")
        print(f"  Recall:    {metrics['sentiment']['recall']:.4f}")
        
        print(f"\nTopic Classification Performance:")
        print(f"  Accuracy:  {metrics['topic']['accuracy']:.4f}")
        print(f"  F1-Score:  {metrics['topic']['f1']:.4f}")
        print(f"  Precision: {metrics['topic']['precision']:.4f}")
        print(f"  Recall:    {metrics['topic']['recall']:.4f}")

        return metrics


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
        # data_path='../data/du_lieu_ao.csv',
        tokenizer=tokenizer,
        max_length=256
    )

    test_loader = create_dataloader(
        test_dataset,
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

    # checkpoint_path = 'checkpoints/t5_model_label.pth'
    # model = T5Tester.load_model(checkpoint_path, device, model_args)

    # tester = T5Tester(model, tokenizer, test_loader, device)
    # tester.test_for_labels()

    checkpoint_path = 'checkpoints/t5_model_topics.pth'
    model = T5Tester.load_model(checkpoint_path, device, model_args)

    tester = T5Tester(model, tokenizer, test_loader, device)
    tester.test_for_topics()

    # test_dataset_combined = T5Dataset_Combined(
    #     data_path='../data/UIT-VSFC/merge_data/test_data.csv',
    #     tokenizer=tokenizer,
    #     max_length=256
    # )

    # test_loader = create_dataloader_combined(
    #     test_dataset_combined,
    #     batch_size=8,
    #     shuffle=False
    # )

    
    # checkpoint_path = 'checkpoints/t5_model_combined.pth'
    # model = T5Tester.load_model(checkpoint_path, device, model_args)

    # tester = T5Tester(model, tokenizer, test_loader, device)
    # tester.test_combined(test_dataloader=test_loader)


if __name__ == "__main__":
    main()