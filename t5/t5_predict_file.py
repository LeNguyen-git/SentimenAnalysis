import pandas as pd
import torch
from t5_model import T5Model, ModelArgs
from t5_tokenizer import T5Tokenizer
from t5_predict import T5Predictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_path = '../data/UIT-VSFC/t5_vocab.json'
checkpoint_path_label = 'checkpoints/t5_model_1.pth'
checkpoint_path_topic = 'checkpoints/t5_model_topics.pth'
max_length = 256

tokenizer = T5Tokenizer()
tokenizer.load_vocab(vocab_path)

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

model_labels = T5Predictor.load_model(checkpoint_path_label, device, model_args)
predict_labels = T5Predictor(model_labels, tokenizer, device)

model_topics = T5Predictor.load_model(checkpoint_path_topic, device, model_args)
predict_topics = T5Predictor(model_topics, tokenizer, device)

def process_file(input, output):
    
    df = pd.read_csv(input)

    results = []

    if "text" in df.columns:
        text_series = df["text"]
    else:
        first_col = df.columns[0]
        text_series = df[first_col]

    for text in text_series:
        sentiment_result = predict_labels.predict_with_label(text)
        topic_result = predict_topics.predict_with_topic(text)

        results.append({
            "text": text,
            "predicted_sentiment": sentiment_result["sentiment"],
            "predicted_topic": topic_result["topic"]
        })
    
    result_df = pd.DataFrame(results)

    # Lưu ra file CSV mới
    result_df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Đã lưu kết quả dự đoán vào {output}")

    # ===== Thống kê sentiment cao nhất cho mỗi topic =====
    summary = result_df.groupby("predicted_topic")["predicted_sentiment"].agg(lambda x: x.value_counts().idxmax())
    print("\n📊 Sentiment phổ biến nhất cho từng topic:")
    print(summary)

if __name__ == "__main__":
    input_csv = input("Nhập đường dẫn file CSV: ")
    output_csv = "../data/predicted_results.csv"
    process_file(input_csv, output_csv)

