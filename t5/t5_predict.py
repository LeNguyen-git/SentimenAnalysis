# import torch
# import re
# from model import T5Model, ModelArgs
# from tokenizer import T5Tokenizer
# from preprocessing import T5Preprocessing
# from mapping_data import sentiment, reverse_sentiment, topic, reverse_topic, reverse_topic_fuzzy
# import json

# import torch.nn.functional as F

# sentiment_labels = [sentiment(i) for i in range(3)] 
# topic_labels = [topic(i) for i in range(4)]


# class T5Predictor:
#     def __init__(self, model, tokenizer, device=None, max_length=256):
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = model.to(self.device)
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.preprocessor = T5Preprocessing()
#         self.model.eval()
    
#     def predict_with_label(self, input_text):
#         preprocessor = T5Preprocessing()
#         processed_text = preprocessor.preprocess_text(input_text)

#         input_text = f"Phân tích cảm xúc cho đoạn văn bản sau: {processed_text}"

#         input_ids = self.tokenizer.encode(input_text, add_special_tokens=True)
#         input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)

#         decode_input_ids = torch.tensor([[self.tokenizer.token_to_id["<s>"]]], dtype=torch.long).to(self.device)
#         generated_ids = []

#         with torch.no_grad():
#             for _ in range(self.max_length):
#                 logits = self.model(input_ids, decode_input_ids)
#                 next_token_logits = logits[:, -1, :]
#                 probs = F.softmax(next_token_logits, dim=-1)

#                 next_token_ids = torch.argmax(probs, dim=-1).unsqueeze(0)
#                 generated_ids.append(next_token_ids.item())

#                 if next_token_ids.item() == self.tokenizer.token_to_id["</s>"]:
#                     break

#                 decode_input_ids = torch.cat([decode_input_ids, next_token_ids], dim=-1)
        
#         pred_text = self.tokenizer.decode(generated_ids)
#         pred_text_clean = pred_text.strip().replace("<pad>", "").replace("<s>", "").replace("</s>", "")

#         try:
#             sentiment_idx = reverse_sentiment(pred_text_clean)
#             sentiment_str = sentiment(sentiment_idx)
#         except:
#             sentiment_idx = None
#             sentiment_str = None
        



#         return {
#             "text": pred_text_clean,
#             "sentiment_idx": sentiment_idx,
#             "sentiment": sentiment_str,
#         }

    
#     def predict_with_topic(self, input_text):

#         preprocessor = T5Preprocessing()
#         processed_text = preprocessor.preprocess_text(input_text)

#         input_text = f"Phân tích chủ đề cho đoạn văn bản sau: {processed_text}"

#         input_ids = self.tokenizer.encode(input_text, add_special_tokens=True)
#         input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)

#         decode_input_ids = torch.tensor([[self.tokenizer.token_to_id["<s>"]]], dtype=torch.long).to(self.device)
#         generated_ids = []

#         with torch.no_grad():
#             for _ in range(self.max_length):
#                 logits = self.model(input_ids, decode_input_ids)
#                 next_token_logits = logits[:, -1, :]
#                 probs = F.softmax(next_token_logits, dim=-1)

#                 next_token_ids = torch.argmax(probs, dim=-1).unsqueeze(0)
#                 generated_ids.append(next_token_ids.item())

#                 if next_token_ids.item() == self.tokenizer.token_to_id["</s>"]:
#                     break

#                 decode_input_ids = torch.cat([decode_input_ids, next_token_ids], dim=-1)
        
#         pred_text = self.tokenizer.decode(generated_ids)
#         pred_text_clean = pred_text.strip().replace("<pad>", "").replace("<s>", "").replace("</s>", "")

#         try:
#             topic_idx, topic_str = reverse_topic_fuzzy(pred_text_clean)
#         except:
#             topic_idx, topic_str = None


#         return {
#             "text": pred_text_clean,
#             "topic_idx": topic_idx,
#             "topic": topic_str,
#         }
    
#     @staticmethod
#     def predict_both_separate_models(text_input, predictor_labels, predictor_topics):
#         sentiment_result = predictor_labels.predict_with_label(text_input)
#         topic_result = predictor_topics.predict_with_topic(text_input)

#         return sentiment_result, topic_result

    
    
#     @staticmethod
#     def load_model(checkpoint_path, device, model_args):
#         model = T5Model(model_args).to(device)
#         checkpoint = torch.load(checkpoint_path)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         print(f"Model loaded from {checkpoint_path}")
#         return model


# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     tokenizer = T5Tokenizer()
#     tokenizer.load_vocab("../data/UIT-VSFC/t5_vocab.json")

#     model_args = ModelArgs(
#         vocab_size=len(tokenizer.token_to_id),
#         d_model=256,
#         num_heads=8,
#         num_layers=6,
#         hidden_dim=512,
#         max_len=256,
#         dropout=0.1,
#         pad_idx=tokenizer.token_to_id["<pad>"]
#     )

#     # Load model cảm xúc
#     checkpoint_labels = 'checkpoints/t5_model_1.pth'
#     model_labels = T5Predictor.load_model(checkpoint_labels, device, model_args)
#     predictor_labels = T5Predictor(model_labels, tokenizer, device)

#     # Load model chủ đề
#     checkpoint_topics = 'checkpoints/t5_model_topics.pth'
#     model_topics = T5Predictor.load_model(checkpoint_topics, device, model_args)
#     predictor_topics = T5Predictor(model_topics, tokenizer, device)

#     # Nhập text
#     text_input = input("Nhập đoạn văn bản: ")

#     # Gọi hàm predict chung
#     result_label, result_topic = T5Predictor.predict_both_separate_models(text_input, predictor_labels, predictor_topics)

#     # In kết quả
#     print("\n======================")
#     print(f"Cảm xúc: {result_label['sentiment']}")

#     print("\n======================")
#     print(f"Chủ đề: {result_topic['topic']}")

import torch
import re
from t5_model import T5Model, ModelArgs
from t5_tokenizer import T5Tokenizer
from t5_preprocessing import T5Preprocessing
from t5_mapping_data import sentiment, reverse_sentiment, topic, reverse_topic, reverse_topic_fuzzy
import json
import torch.nn.functional as F
import numpy as np

sentiment_labels = [sentiment(i) for i in range(3)] 
topic_labels = [topic(i) for i in range(4)]


class T5Predictor:
    def __init__(self, model, tokenizer, device=None, max_length=256):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocessor = T5Preprocessing()
        self.model.eval()
        
        self.sentiment_tokens = self._get_label_tokens([sentiment(i) for i in range(3)])
        self.topic_tokens = self._get_label_tokens([topic(i) for i in range(4)])
    
    def _get_label_tokens(self, labels):
        label_tokens = {}
        for i, label in enumerate(labels):
            tokens = self.tokenizer.encode(label, add_special_tokens=False)
            label_tokens[i] = tokens
        return label_tokens 
    
    def _calculate_sequence_probability(self, input_ids, target_sequence):
        decode_input_ids = torch.tensor([[self.tokenizer.token_to_id["<s>"]]], 
                                       dtype=torch.long).to(self.device)
        
        log_prob = 0.0
        with torch.no_grad():
            for target_token in target_sequence:
                logits = self.model(input_ids, decode_input_ids)
                next_token_logits = logits[:, -1, :]
                log_probs = F.log_softmax(next_token_logits, dim=-1)
                
                log_prob += log_probs[0, target_token].item()
                
                target_tensor = torch.tensor([[target_token]], dtype=torch.long).to(self.device)
                decode_input_ids = torch.cat([decode_input_ids, target_tensor], dim=-1)
        
        return log_prob
    
    def _get_all_label_probabilities(self, input_ids, label_tokens_dict, task_type="sentiment"):
        label_probs = {}
        log_probs = []
        
        for label_idx, token_sequence in label_tokens_dict.items():
            full_sequence = token_sequence + [self.tokenizer.token_to_id["</s>"]]
            log_prob = self._calculate_sequence_probability(input_ids, full_sequence)
            label_probs[label_idx] = log_prob
            log_probs.append(log_prob)
        
        log_probs = np.array(log_probs)
        log_probs = log_probs - np.max(log_probs)
        probs = np.exp(log_probs)
        probs = probs / np.sum(probs)
        
        return probs, label_probs

    def predict_with_label_confidence(self, input_text):
        preprocessor = T5Preprocessing()
        processed_text = preprocessor.preprocess_text(input_text)
        
        prompt = f"Phân tích cảm xúc cho đoạn văn bản sau: {processed_text}"
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        probs, label_log_probs = self._get_all_label_probabilities(
            input_ids, self.sentiment_tokens, "sentiment"
        )
        
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]
        
        generated_text = self._generate_text(input_ids)
        
        try:
            generated_sentiment_idx = reverse_sentiment(generated_text)
            generated_sentiment_str = sentiment(generated_sentiment_idx)
        except:
            generated_sentiment_idx = None
            generated_sentiment_str = "Unknown"
        
        label_map = {0: 'Tiêu cực', 1: 'Trung tính', 2: 'Tích cực'}
        
        return {
            'label': label_map.get(pred_idx, 'Unknown'),
            'confidence': round(confidence * 100, 2),
            'probabilities': [round(p * 100, 2) for p in probs],
            'original_text': input_text,
            'preprocessed_text': processed_text,
            'generated_text': generated_text,
            'generated_label': generated_sentiment_str,
            'method_agreement': (pred_idx == generated_sentiment_idx) if generated_sentiment_idx is not None else False
        }
    
    def predict_with_topic_confidence(self, input_text):
        preprocessor = T5Preprocessing()
        processed_text = preprocessor.preprocess_text(input_text)
        
        prompt = f"Phân tích chủ đề cho đoạn văn bản sau: {processed_text}"
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        probs, label_log_probs = self._get_all_label_probabilities(
            input_ids, self.topic_tokens, "topic"
        )
        
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]
        
        generated_text = self._generate_text(input_ids)
        
        try:
            generated_topic_idx, generated_topic_str = reverse_topic_fuzzy(generated_text)
        except:
            generated_topic_idx, generated_topic_str = None, "Unknown"
        
        topic_map = {0: "Giảng viên", 1: "Chương trình học", 2: "Cơ sở vật chất", 3: "Khác"}
        
        return {
            'topic': topic_map.get(pred_idx, 'Unknown'),
            'confidence': round(confidence * 100, 2),
            'probabilities': [round(p * 100, 2) for p in probs],
            'original_text': input_text,
            'preprocessed_text': processed_text,
            'generated_text': generated_text,
            'generated_topic': generated_topic_str,
            'method_agreement': (pred_idx == generated_topic_idx) if generated_topic_idx is not None else False
        }
    
    def _generate_text(self, input_ids):
        decode_input_ids = torch.tensor([[self.tokenizer.token_to_id["<s>"]]], 
                                       dtype=torch.long).to(self.device)
        generated_ids = []
        
        with torch.no_grad():
            for _ in range(self.max_length):
                logits = self.model(input_ids, decode_input_ids)
                next_token_logits = logits[:, -1, :]
                next_token_ids = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                generated_ids.append(next_token_ids.item())
                
                if next_token_ids.item() == self.tokenizer.token_to_id["</s>"]:
                    break
                    
                decode_input_ids = torch.cat([decode_input_ids, next_token_ids], dim=-1)
        
        pred_text = self.tokenizer.decode(generated_ids)
        pred_text_clean = pred_text.strip().replace("<pad>", "").replace("<s>", "").replace("</s>", "")
        return pred_text_clean
    
    @staticmethod
    def predict_both_with_confidence(text_input, predictor_labels, predictor_topics):
        sentiment_result = predictor_labels.predict_with_label_confidence(text_input)
        topic_result = predictor_topics.predict_with_topic_confidence(text_input)
        
        return sentiment_result, topic_result
    
    @staticmethod
    def load_model(checkpoint_path, device, model_args):
        model = T5Model(model_args).to(device)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {checkpoint_path}")
        return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = T5Tokenizer()
    tokenizer.load_vocab("../data/UIT-VSFC/t5_vocab.json")

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

    # Load models
    checkpoint_labels = 'checkpoints/t5_model_label.pth'
    model_labels = T5Predictor.load_model(checkpoint_labels, device, model_args)
    predictor_labels = T5Predictor(model_labels, tokenizer, device)

    checkpoint_topics = 'checkpoints/t5_model_topics.pth'
    model_topics = T5Predictor.load_model(checkpoint_topics, device, model_args)
    predictor_topics = T5Predictor(model_topics, tokenizer, device)

    while True:
        text_input = input("Nhập văn bản để dự đoán ('exit' hoặc 'thoát' để kết thúc): ")
        if text_input.lower() in ['exit', 'thoát']:
            break
            
        if not text_input.strip():
            print("Vui lòng nhập một văn bản hợp lệ.")
            continue

        print("\n" + "="*50)
        print("Tính xác suất cho tất cả labels")
        
        sentiment_result, topic_result = T5Predictor.predict_both_with_confidence(
            text_input, predictor_labels, predictor_topics
        )
        
        print(f"Văn bản: {text_input}")
        print(f"\n--- SENTIMENT ---")
        print(f"Dự đoán: {sentiment_result['label']} (Độ tin cậy: {sentiment_result['confidence']}%)")
        print(f"Xác suất: Tiêu cực: {sentiment_result['probabilities'][0]}%, Trung tính: {sentiment_result['probabilities'][1]}%, Tích cực: {sentiment_result['probabilities'][2]}%")
        print(f"Generated text: {sentiment_result['generated_text']}")
        print(f"Method agreement: {sentiment_result['method_agreement']}")
        
        print(f"\n--- TOPIC ---")
        print(f"Dự đoán: {topic_result['topic']} (Độ tin cậy: {topic_result['confidence']}%)")
        print(f"Xác suất: Giảng viên: {topic_result['probabilities'][0]}%, Chương trình học: {topic_result['probabilities'][1]}%, Cơ sở vật chất: {topic_result['probabilities'][2]}%, Khác: {topic_result['probabilities'][3]}%")
        print(f"Generated text: {topic_result['generated_text']}")
        print(f"Method agreement: {topic_result['method_agreement']}")

        
        print("-" * 50)