import torch
import torch.nn as nn
from model import MiniLlamaModel, ModelArgs
from tokenizer import LLaMaTokenizer
import preprocessing

def predict(model, tokenizer, text, device, max_length=128):
    model.eval()

    processed_text = preprocessing.preprocess_text(text)
    token_ids = tokenizer.encode(processed_text, add_special_tokens=True)

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

    return label_map[pred], confidence, probs[0].tolist()

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
    tokenizer.load_vocab('../data/UIT-VSFC/vocab.json')

    model_args = ModelArgs(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=256,
        d_model=256,
        n_layers=6,
        n_heads=8,
        max_seq_len=128,
        num_classes=3
    )

    checkpoint_path = 'checkpoints/model_2.pth'
    model = load_model(checkpoint_path, device, model_args)

    while True:
        text = input("Nhập câu để dự đoán cảm xúc (hoặc 'thoát' để thoát): ")
        if text.lower() == 'thoát' or text.lower() == 'exit':
            print("Kết thúc chương trình.")
            break

        if not text.strip():
            print("Vui lòng nhập một câu hợp lệ!")
            continue

        label, confidence, probs = predict(model, tokenizer, text, device)
        print(f"Câu: {text}")
        print(f"Dự đoán: {label} (Độ tin cậy: {confidence:.4f})")
        print(f"Xác suất: Tiêu cực: {probs[0]:.4f}, Trung tính: {probs[1]:.4f}, Tích cực: {probs[2]:.4f}\n")

if __name__ == "__main__":
    main()