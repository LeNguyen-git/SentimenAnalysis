import torch
import json
from bert_tokenizer import BertTokenizer
from bert_model import BertModel, ModelArgs
from bert_dataset import BertDataset, create_data_loader
from bert_preprocessing import TextPreprocessor
from tqdm import tqdm
from pyvi import ViTokenizer


def predict_sentiment(model, tokenizer, preprocessor, text, device, max_length=256):
    model.eval()

    preprocessed_text = preprocessor.preprocess_text(text)

    # token_ids = tokenizer.encode(ViTokenizer.tokenize(preprocessed_text), add_special_tokens=True)
    token_ids = tokenizer.encode(preprocessed_text, add_special_tokens = True)

    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length] + [tokenizer.token_to_id('[SEP]')]
    else:
        token_ids.extend([tokenizer.token_to_id('[PAD]')] * (max_length - len(token_ids)))

    attention_mask = [1 if token_id != tokenizer.token_to_id('[PAD]') else 0 for token_id in token_ids]
    token_type_ids = [0] * max_length

    input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)
    token_type_ids = torch.tensor([token_type_ids], dtype=torch.long).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs['logits']
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(logits, dim=-1).cpu().item()
        prob = probs.cpu().numpy()[0]
    
    label_map = {0: 'Tiêu cực', 1: 'Trung tính', 2: 'Tích cực'}
    confidence = probs[0][pred].item()

    return {
        'label': label_map[pred],
        'confidence': round(confidence * 100, 2),  
        'probabilities': [round(p * 100, 2) for p in probs[0].tolist()],
        'original_text': text,
        'preprocessed_text': preprocessed_text
    }


def predict_topic(model, tokenizer, preprocessor, text, device, max_length=256):
    model.eval()

    preprocessed_text = preprocessor.preprocess_text(text)

    # token_ids = tokenizer.encode(ViTokenizer.tokenize(preprocessed_text), add_special_tokens=True)
    token_ids = tokenizer.encode(preprocessed_text, add_special_tokens = True)

    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length] + [tokenizer.token_to_id('[SEP]')]
    else:
        token_ids.extend([tokenizer.token_to_id('[PAD]')] * (max_length - len(token_ids)))

    attention_mask = [1 if token_id != tokenizer.token_to_id('[PAD]') else 0 for token_id in token_ids]
    token_type_ids = [0] * max_length

    input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)
    token_type_ids = torch.tensor([token_type_ids], dtype=torch.long).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs['topic_logits']
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(logits, dim=-1).cpu().item()
        prob = probs.cpu().numpy()[0]
    
    topic_map = {0: "Giảng viên", 1: "Chương trình học", 2: "Cơ sở vật chất", 3: "Khác"}
    confidence = probs[0][pred].item()

    return {
        'topic': topic_map[pred],
        'confidence': round(confidence * 100, 2),  # Độ tin cậy %
        'probabilities': [round(p * 100, 2) for p in probs[0].tolist()],
        'original_text': text,
        'preprocessed_text': preprocessed_text
    }


def load_model(checkpoint_path, device, model_args, num_labels=None, num_topics=None):
    model = BertModel(model_args, num_labels=num_labels, num_topics=num_topics).to(device) 
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print(f"Mô hình đã được load từ  {checkpoint_path}")
    return model


def init_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer(min_frequency=1)
    tokenizer.load_vocab('../data/UIT-VSFC/bert_vocab.json')
    preprocessor = TextPreprocessor()

    model_args_label = ModelArgs(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_layers=6,
        n_heads=8,
        hidden_dim=512,
        max_positions=512,
        type_vocab_size=2,
        dropout=0.1
    )
    model_label = load_model('../bert/checkpoints/bert_model_label.pth', device, model_args_label, num_labels=3)

    model_args_topic = ModelArgs(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_layers=6,
        n_heads=8,
        hidden_dim=512,
        max_positions=512,
        type_vocab_size=2,
        dropout=0.1
    )
    model_topic = load_model('../bert/checkpoints/bert_model_topic.pth', device, model_args_topic, num_topics=4)

    return model_label, model_topic, tokenizer, preprocessor, device


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    tokenizer = BertTokenizer(min_frequency=1)
    tokenizer.load_vocab('../data/UIT-VSFC/bert_vocab.json')
    preprocessor = TextPreprocessor()

    model_args_label = ModelArgs(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_layers=6,
        n_heads=8,
        hidden_dim=512,
        max_positions=512,
        type_vocab_size=2,
        dropout=0.1
    )

    checkpoint_path_label = 'checkpoints/bert_model_label.pth'
    model_label = load_model(checkpoint_path_label, device, model_args_label, num_labels=3)

    model_args_topic = ModelArgs(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_layers=6,
        n_heads=8,
        hidden_dim=512,
        max_positions=512,
        type_vocab_size=2,
        dropout=0.1
    )

    checkpoint_path_topic = 'checkpoints/bert_model_topic.pth'
    model_topic = load_model(checkpoint_path_topic, device, model_args_topic, num_topics=4)

    while True:
        text = input("Nhập văn bản để dự đoán ('exit' hoặc 'thoát' để kết thúc): ")
        if text.lower() == 'exit' or text.lower() == 'thoát':
            break

        if not text.strip():
            print("Vui lòng nhập một văn bản hợp lệ.")
            continue

        sentiment_result = predict_sentiment(model_label, tokenizer,preprocessor, text,device, max_length=256)
        topic_result = predict_topic(model_topic, tokenizer, preprocessor, text, device, max_length=256)

        label_sentiment = sentiment_result['label']
        confidence_sentiment = sentiment_result['confidence']
        probs_sentiment = sentiment_result['probabilities']

        label_topics = topic_result['topic']
        confidence_topics = topic_result['confidence']
        probs_topics = topic_result['probabilities']


 
        print(f"Văn bản: {text}")
        print(f"Dự đoán: {label_sentiment} (Độ tin cậy: {confidence_sentiment:.4f})")
        print(f"Dự đoán: Tích cực: {probs_sentiment[2]:.4f}, Trung tính: {probs_sentiment[1]:.4f}, Tiêu cực: {probs_sentiment[0]:.4f}\n")

        print(f"Dự đoán: {label_topics} (Độ tin cậy: {confidence_topics:.4f})")
        print(f"Xác suất: Giảng viên: {probs_topics[0]:.4f}, Chương trình học: {probs_topics[1]:.4f}, Cơ sở vật chất: {probs_topics[2]:.4f}, Khác: {probs_topics[3]:.4f}\n")



        print("-" * 50)

if __name__ == "__main__":
    main()
