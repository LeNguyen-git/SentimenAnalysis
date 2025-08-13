import torch
import torch.nn as nn
from llama_model import MiniLlamaModel, ModelArgs
from llama_tokenizer import LLaMaTokenizer
import llama_preprocessing
from pyvi import ViTokenizer

def predict_for_sentiment(model, tokenizer, text, device, max_length=128):
    model.eval()

    processed_text = llama_preprocessing.preprocess_text(text)
    token_ids = tokenizer.encode(processed_text, add_special_tokens=True)
    # token_ids = tokenizer.encode(ViTokenizer.tokenize(processed_text), add_special_tokens=True)

    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length-1] + [tokenizer.str_to_int['</s>']]
    else:
        pad_length = max_length - len(token_ids)
        token_ids = token_ids + [tokenizer.str_to_int['<pad>']] * pad_length

    attention_mask = [1 if token_id != tokenizer.str_to_int['<pad>'] else 0 for token_id in token_ids]

    input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        logits = outputs['logits']
        probs = torch.softmax(logits, dim=-1)  
        pred = torch.argmax(logits, dim=-1).item()

    label_map = {0: "Tiêu cực", 1: "Trung tính", 2: "Tích cực"}
    confidence = probs[0][pred].item()

    return {
        'label': label_map[pred],
        'confidence': confidence,
        'probabilities': probs[0].tolist(),
        'original_text': text,
        'preprocessed_text': processed_text
    }

def predict_for_topic(model, tokenizer, text, device, max_length=128):
    model.eval()

    processed_text = llama_preprocessing.preprocess_text(text)
    token_ids = tokenizer.encode(processed_text, add_special_tokens=True)
    # token_ids = tokenizer.encode(ViTokenizer.tokenize(processed_text), add_special_tokens=True)

    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length-1] + [tokenizer.str_to_int['</s>']]
    else:
        pad_length = max_length - len(token_ids)
        token_ids = token_ids + [tokenizer.str_to_int['<pad>']] * pad_length

    attention_mask = [1 if token_id != tokenizer.str_to_int['<pad>'] else 0 for token_id in token_ids]

    input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        logits = outputs['topic_logits']
        probs = torch.softmax(logits, dim=-1)  
        pred = torch.argmax(logits, dim=-1).item()

    topic_map = {0: "Giảng viên", 1: "Chương trình học", 2: "Cơ sở vật chất", 3: "Khác"}
    confidence = probs[0][pred].item()

    return {
        'topic': topic_map[pred],
        'confidence': confidence,
        'probabilities': probs[0].tolist(),
        'original_text': text,
        'preprocessed_text': processed_text
    }

def load_model(checkpoint_path, device, model_args):

    model = MiniLlamaModel(model_args).to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Mô hình load từ {checkpoint_path}")
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    tokenizer = LLaMaTokenizer({})
    tokenizer.load_vocab('../data/UIT-VSFC/llama_vocab.json')

    model_args_sentiment = ModelArgs(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=512,
        d_model=256,
        n_layers=6,
        n_heads=8,
        max_seq_len=128,
        num_labels=3,
        num_groups=4
    )

    
    checkpoint_path_sentiment = 'checkpoints/llama_model_label.pth'
    model_sentiment = load_model(checkpoint_path_sentiment, device, model_args_sentiment)

    model_args_topics = ModelArgs(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=512,
        d_model=256,
        n_layers=6,
        n_heads=8,
        max_seq_len=128,
        num_topics=4,
        num_groups=4
    )

    
    checkpoint_path_topic = 'checkpoints/llama_model_topic.pth'
    model_topic = load_model(checkpoint_path_topic, device, model_args_topics)

    while True:
        text = input("Nhập câu để dự đoán cảm xúc (hoặc 'thoát' để thoát): ")
        if text.lower() == 'thoát' or text.lower() == 'exit':
            print("Kết thúc chương trình.")
            break

        if not text.strip():
            print("Vui lòng nhập một câu hợp lệ!")
            continue

        sentiment_result = predict_for_sentiment(model_sentiment, tokenizer, text, device)
        topic_result = predict_for_topic(model_topic, tokenizer, text, device)

        label_sentiment = sentiment_result['label']
        confidence_sentiment = sentiment_result['confidence']
        probs_sentiment = sentiment_result['probabilities']

        label_topics = topic_result['topic']
        confidence_topics = topic_result['confidence']
        probs_topics = topic_result['probabilities']


        print(f"Câu: {text}")
        print(f"Dự đoán: {label_sentiment} (Độ tin cậy: {confidence_sentiment}%)")
        print(f"Dự đoán: Tích cực: {probs_sentiment[2]}%, Trung tính: {probs_sentiment[1]}%, Tiêu cực: {probs_sentiment[0]}%\n")

        print(f"Dự đoánL {label_topics} (Độ tin cậy: {confidence_topics}%)")
        print(f"Xác suất: Giảng viên: {probs_topics[0]}%, Chương trình học: {probs_topics[1]}%, Cơ sở vật chất: {probs_topics[2]}%, Khác: {probs_topics[3]}%\n")


if __name__ == "__main__":
    main()