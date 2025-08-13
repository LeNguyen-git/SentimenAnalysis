#Chỉ số -> Nhãn cảm xúc
def sentiment(label: int) -> str:
    mapping = {
        0: "Tiêu cực",
        1: "Trung tính",
        2: "Tích cực"
    }
    return mapping.get(label, "Trung tính")

#Chỉ số -> Nhãn chủ đề
def topic(topic: int) -> str:
    mapping = {
        0: "Giảng viên",
        1: "Chương trình học",
        2: "Cơ sở vật chất",
        3: "Khác"
    }
    return mapping.get(topic, "Khác")  

# Nhãn cảm xúc -> Chỉ số
def reverse_sentiment(label: str) -> int:
    mapping = {
        "Tiêu cực": 0,
        "Trung tính": 1,
        "Tích cực": 2,
        "tiêu cực": 0,
        "trung tính": 1,
        "tích cực": 2,
        "tích tính": 2,
        "tiêu tính": 0,
        "trung cực": 1
    }
    cleaned_label = label.strip().lower().split("</s>")[0].strip()
    return mapping.get(cleaned_label, 1)

#Nhãn chủ đề -> Chỉ số
def reverse_topic(label: str) -> int:
    mapping = {
        "Giảng viên": 0,
        "Chương trình học": 1,
        "Cơ sở vật chất": 2,
        "Khác": 3,
        "giảng viên": 0,
        "chương trình học": 1,
        "cơ sở vật chất": 2,
        "khác": 3
    }
    cleaned_label = label.strip().lower().split("</s>")[0].strip()
    return mapping.get(cleaned_label, 3)

from rapidfuzz import process


SENTIMENT_CLASSES = ["Tiêu cực", "Trung tính", "Tích cực"]
TOPIC_CLASSES = ["Giảng viên", "Chương trình học", "Cơ sở vật chất", "Khác"]

def reverse_sentiment_fuzzy(label: str, threshold=60) -> int:
    cleaned_label = label.strip().lower().split("</s>")[0].strip()
    match, score, idx = process.extractOne(cleaned_label, [c.lower() for c in SENTIMENT_CLASSES])
    if score >= threshold:
        return SENTIMENT_CLASSES[idx]
    return "Không rõ"


def reverse_topic_fuzzy(label: str, threshold=60) -> int:
    cleaned_label = label.strip().lower().split("</s>")[0].strip()
    match, score, idx = process.extractOne(cleaned_label, [c.lower() for c in TOPIC_CLASSES])
    if score >= threshold:
        return  idx, TOPIC_CLASSES[idx]
    return -1, "Không rõ"





# if __name__ == "__main__":
#     import pandas as pd

#     # Đọc dữ liệu mẫu
#     data = pd.read_csv("../data/UIT-VSFC/merge_data/dev_data.csv")

#     print("Dữ liệu gốc:")
#     print(data.head())

#     # Mapping chỉ số thành chuỗi
#     data["sentiment"] = data["sentiment"].apply(sentiment)
#     data["topic"] = data["topic"].apply(topic)

#     print("\nDữ liệu sau khi ánh xạ:")
#     print(data.head())

#     # Thống kê số lượng mỗi chủ đề
#     print("\nThống kê số lượng từng chủ đề:")
#     print(data["topic"].value_counts())

#     print(f"\nTổng số chủ đề khác nhau: {data['topic'].nunique()}")
