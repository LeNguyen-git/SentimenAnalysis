import argparse
import os
import pandas as pd
from collections import defaultdict, Counter

from lime.lime_text import LimeTextExplainer
from pyvi import ViTokenizer

from bert_predict import init_models, predict_sentiment, predict_topic
from bert_preprocessing import TextPreprocessor

class_names = ["Tiêu cực", "Trung tính", "Tích cực"]
topic_names = ["Giảng viên", "Chương trình học", "Cơ sở vật chất", "Khác"]
STOPWORDS = set(["không", "rất", "quá", "này", "đó", "có", "còn", "và", "nhưng",
                  "thì", "đã", "sẽ", "là", "của", "cho", "với", "trong", "khi", 
                  "được", "và", 
                ])

explainer = LimeTextExplainer(class_names=class_names)

model_label, model_topic, tokenizer, preprocessor, device = init_models()


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

def predict_file(input_path, model_label=None, model_topic=None, tokenizer=None, preprocessor=None, device=None, output_path=None, save=False):
    if model_label is None or model_topic is None:
        model_label, model_topic, tokenizer, preprocessor, device = init_models()

    data = load_input_file(input_path)

    results = []
    stats = defaultdict(lambda: defaultdict(int))

    for text in data["text"]:
        pred_sentiment = predict_sentiment(model_label, tokenizer, preprocessor, text, device)
        sentiment_label = pred_sentiment["label"]

        pred_topic = predict_topic(model_topic, tokenizer, preprocessor, text, device)
        topic_label = pred_topic["topic"]

        stats[topic_label][sentiment_label] += 1

        results.append({
            "predicted_topic": topic_label,
            "text": text,
            "predicted_sentiment": sentiment_label
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

def lime_predict_proba(texts):
    import torch
    import numpy as np

    results = []
    for text in texts:
        pre_text = preprocessor.preprocess_text(text)
        token_ids = tokenizer.encode(ViTokenizer.tokenize(pre_text), add_special_tokens=True)
        token_ids = token_ids[:256]
        attention_mask = [1] * len(token_ids)
        token_type_ids = [0] * len(token_ids)

        input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)
        token_type_ids = torch.tensor([token_type_ids], dtype=torch.long).to(device)

        with torch.no_grad():
            outputs = model_label(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
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




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dự đoán cảm xúc & chủ đề từ file")
    parser.add_argument("input_path", help="Đường dẫn file input (.txt, .csv, .json, .xls, .xlsx)")
    parser.add_argument("--save", action="store_true", help="Lưu kết quả ra file CSV")
    parser.add_argument("--output", help="Đường dẫn file kết quả (mặc định: cùng thư mục input)")
    args = parser.parse_args()

    output_path = args.output
    if args.save and not output_path:
        base, ext = os.path.splitext(args.input_path)
        output_path = f"{base}_result.csv"

    results, stats = predict_file(
        input_path=args.input_path,
        output_path=output_path,
        save=args.save
    )

    print("\n📊 Thống kê theo chủ đề:")
    for topic, info in stats.items():
        print(f"- {topic}: Cảm xúc cao nhất: {info['max_sentiment']}")
        for sentiment, pct in info["percentages"].items():
            print(f"   {sentiment}: {pct}%")

    print("\n📝 Kết luận LIME:")
    for target_sentiment in class_names:
        explanations = summarize_sentiment_reasons_by_topic(results, target_sentiment=target_sentiment)
        for line in explanations:
            print("-", line)
        print("=" * 50)