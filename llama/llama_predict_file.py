import os
import torch
import pandas as pd
import numpy as np

from collections import defaultdict, Counter
from llama_predict import predict_for_sentiment, predict_for_topic
from llama_preprocessing import preprocess_text
from llama_tokenizer import LLaMaTokenizer
from llama_model import MiniLlamaModel, ModelArgs



from lime.lime_text import LimeTextExplainer
from pyvi import ViTokenizer


def load_input_file(input_path):
    ext = os.path.splitext(input_path)[-1].lower()

    if ext == ".txt":
        with open(input_path, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        return pd.DataFrame({"text": texts})

    elif ext == ".csv":
        df = pd.read_csv(input_path)
    elif ext == ".json":
        df = pd.read_json(input_path, encoding="utf-8")
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(input_path)
    else:
        raise ValueError("Chỉ hỗ trợ .txt, .csv, .json, .xls, .xlsx")

    if "text" not in df.columns:
        first_str_col = None
        for col in df.columns:
            if df[col].dtype == object:
                first_str_col = col
                break
        if first_str_col is None:
            raise ValueError("Không tìm thấy cột văn bản")
        df = df.rename(columns={first_str_col: "text"})

    return df

def predict_file(input_path, output_path, model, tokenizer, device, save = False):

    data = load_input_file(input_path)

    results = []
    stats = defaultdict(lambda: defaultdict(int))

    for text in data["text"]:
        pred_sentiment = predict_for_sentiment(model, tokenizer, text, device)
        sentiment_label = pred_sentiment["sentiment"]

        pred_topic = predict_for_topic(model, tokenizer, text, device)
        topic_label = pred_topic["topic"]

        stats[topic_label][sentiment_label] += 1

        results.append({
            "text": text,
            "predicted_sentiment": sentiment_label,
            "predicted_topic": topic_label
        })

    if save and output_path:
        if not output_path.lower().endswith(".csv"):
            output_path += ".csv"
        results_data = pd.DataFrame(results)
        results_data.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Kết quả đã lưu vào {output_path}")
    
    stats_percent = {}
    for topic, sentiment_counts in stats.items():
        total = sum(sentiment_counts.values())
        percent_dict = {
            sentiment: round(count / total * 100, 2)
            for sentiment, count in sentiment_counts.items()
        }
        max_sentiment = max(sentiment_counts, key=sentiment_counts.get)
        stats_percent[topic] = {
            "max_sentiment": max_sentiment,
            "percentages": percent_dict,
            "counts": dict(sentiment_counts),
            "total": total,
        }

    return results, stats_percent   

class_names = ["Tiêu cực", "Trung tính", "Tích cực"]
topic_names = ["Giảng viên", "Chương trình học", "Cơ sở vật chất", "Khác"]
STOPWORDS = set(["không", "rất", "quá", "này", "đó", "có", "còn", "và", "nhưng",
                  "thì", "đã", "sẽ", "là", "của", "cho", "với", "trong", "khi", 
                  "được", "và", 
                ])

explainer = LimeTextExplainer(class_names=class_names)


def lime_predict_proba(texts, model, device):
    results = []
    for text in texts:
        pre_text = preprocess_text(text)

        token_ids = LLaMaTokenizer.encode(ViTokenizer.tokenize(pre_text), add_special_tokens=True)
        token_ids = token_ids[:256]
        attention_mask = [1] * len(token_ids)
        token_type_ids = [0] * len(token_ids)

        input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)
        token_type_ids = torch.tensor([token_type_ids], dtype=torch.long).to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        results.append(probs[0])
    return np.array(results)

def summarize_sentiment_reasons_by_topic(results, target_sentiment=None, target_topic=None):
    topic_word_map = defaultdict(list)

    for row in results:
        sentiment = row["predicted_sentiment"]
        topic = row["predicted_topic"]

        if sentiment != target_sentiment:
            continue
        if target_topic and topic != target_topic:
            continue

        text = row["text"]

        label_idx = class_names.index(sentiment)
        exp = explainer.explain_instance(
            ViTokenizer.tokenize(text),
            lime_predict_proba,
            num_features=15,
            labels=[label_idx],
            num_samples=60
        )

        for word, weight in exp.as_list(label=label_idx):
            word = word.strip().lower()
            if weight > 0 or word not in STOPWORDS:
                topic_word_map[topic].append(word)

    conclusions = {}
    for topic, words in topic_word_map.items():
        counter = Counter(words)
        top_words = [w for w, _ in counter.most_common(10)]
        word_str = ", ".join(top_words)
        conclusions[topic] = (
            f"Do các bình luận thuộc chủ đề **{topic}** "
            f"thường xuất hiện các từ như **{word_str}**, "
            f"nên sẽ mang thiên hướng **{target_sentiment}**."
        )

    return conclusions

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
    from llama_predict import load_model
    model_sentiment = load_model(checkpoint_path_sentiment, device, model_args_sentiment)

    model_args_topic = ModelArgs(
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
    model_topic = load_model(checkpoint_path_topic, device, model_args_topic)

    input_path = "../demo/uploads/quicktest.json"    
    output_path = "output_results.csv"

    results = []
    stats = defaultdict(lambda: defaultdict(int))
    data = load_input_file(input_path)

    if 'text' not in data.columns:
        raise ValueError("File đầu vào phải có cột 'text'")

    for text in data['text']:
        pred_sentiment = predict_for_sentiment(model_sentiment, tokenizer, text, device)
        pred_topic = predict_for_topic(model_topic, tokenizer, text, device)

        sentiment_label = pred_sentiment['label']
        topic_label = pred_topic['topic']

        stats[topic_label][sentiment_label] += 1

        results.append({
            'text': text,
            'predicted_sentiment': sentiment_label,
            'sentiment_confidence': pred_sentiment['confidence'],
            'predicted_topic': topic_label,
            'topic_confidence': pred_topic['confidence']
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Kết quả dự đoán sentiment và topic đã lưu tại {output_path}")

    print("\n=== Thống kê tổng quan ===")
    for topic, sentiment_counts in stats.items():
        total = sum(sentiment_counts.values())
        percentages = {s: round(c / total * 100, 2) for s, c in sentiment_counts.items()}
        max_sentiment = max(sentiment_counts, key=sentiment_counts.get)
        print(f"Chủ đề: {topic}")
        print(f"  Tổng bình luận: {total}")
        print(f"  Phân bố cảm xúc: {percentages}")
        print(f"  Cảm xúc chiếm đa số: {max_sentiment}\n")

if __name__ == "__main__":
    main()
