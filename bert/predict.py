import torch
import json
from tokenizer import BertTokenizer
from model import BertModel, ModelArgs
from dataset import BertDataset, create_data_loader
from preprocessing import TextPreprocessor
from tqdm import tqdm

def predict(model, tokenizer, preprocessor, text, device, max_length=256):
    model.eval()

    preprossed_text = preprocessor.preprocess_text(text)
    token_ids = tokenizer.encode(preprossed_text, add_special_tokens=True)

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
    return {
        'text': text,
        'processed_text': preprossed_text,
        'prediction': label_map[pred],
        'probabilities': prob.tolist(),
        'label_id': pred
    }

def load_model(checkpoint_path, device, model_args, num_labels=3):
    model = BertModel(model_args, num_labels=num_labels).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Mô hình đã được load từ {checkpoint_path}")
    return model


def main():
    divece = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {divece}")

    tokenizer = BertTokenizer(min_frequency=1)
    tokenizer.load_vocab('../data/UIT-VSFC/bert_vocab.json')
    preprocessor = TextPreprocessor()

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

    checkpoint_path = 'checkpoints/bert_model.pth'
    model = load_model(checkpoint_path, divece, model_args, num_labels=3)

    while True:
        text = input("Nhập văn bản để dự đoán ('exit' hoặc 'thoát' để kết thúc): ")
        if text.lower() == 'exit' or text.lower() == 'thoát':
            break

        if not text.strip():
            print("Vui lòng nhập một văn bản hợp lệ.")
            continue

        prediction = predict(model, tokenizer, preprocessor, text, divece, max_length=256)
        print(f"Văn bản: {prediction['text']}")
        print(f"Văn bản đã xử lý: {prediction['processed_text']}")
        print(f"Dự đoán: {prediction['prediction']} (ID: {prediction['label_id']})")
        print(f"Xác suất: {prediction['probabilities']}")
        print("-" * 50)

if __name__ == "__main__":
    main()
