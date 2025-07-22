#Chỉ số -> Nhãn cảm xúc
def sentiment(label: int) -> str:
    mapping = {
        0: "Tiêu cực",
        1: "Trung tính",
        2: "Tích cực"
    }
    return mapping.get(label, "Trung tính")

#Chỉ số -> Nhãn chủ đề
def topic(label: int) -> str:
    mapping = {
        0: "Giảng viên",
        1: "Chương trình học",
        2: "Cơ sở vật chất",
        3: "Khác"
    }
    return mapping.get(label, "Khác")  

#Nhãn cảm xúc -> Chỉ số
def reverse_sentiment(label: str) -> int:
    mapping = {
        "Tiêu cực": 0,
        "Trung tính": 1,
        "Tích cực": 2
    }
    return mapping.get(label.strip(), 1)  

#Nhãn chủ đề -> Chỉ số
def reverse_topic(label: str) -> int:
    mapping = {
        "Giảng viên": 0,
        "Chương trình học": 1,
        "Cơ sở vật chất": 2,
        "Khác": 3
    }
    return mapping.get(label.strip(), 3)


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
