# import os
# from dotenv import load_dotenv
# import google.generativeai as genai

# load_dotenv()
# api_key = os.getenv("GEMINI_API_KEY")

# if not api_key:
#     raise ValueError("⚠️ Không tìm thấy GEMINI_API_KEY trong file .env")

# genai.configure(api_key=api_key)

# def check_and_explain_gemini(text, predicted_sentiment, predicted_topic):
#     prompt = f"""
#             Bạn là chuyên gia phân tích văn bản tiếng Việt.
#             Dưới đây là một bình luận và kết quả dự đoán từ mô hình:
#             Bình luận: "{text}"
#             Dự đoán cảm xúc: {predicted_sentiment}
#             Dự đoán chủ đề: {predicted_topic}

#             Yêu cầu:
#             1. Kiểm tra xem dự đoán cảm xúc và chủ đề có hợp lý không.
#             2. Nếu thấy sai, hãy đưa ra dự đoán đúng hơn.
#             3. Giải thích ngắn gọn tại sao văn bản ( hoặc nhiều đoạn văn bản) này lại có cảm xúc đó. Liệt kê vài từ có liên quan đến cảm xúc đó.
#             4. Giải thích ngắn gọn tại sao văn bản ( hoặc nhiều đoạn văn bản) này lại có chủ đề đó.
#             Trả lời dưới dạng JSON với các trường:
#             - sentiment
#             - topic
#             - explanation
#             """

#     # model = genai.GenerativeModel("gemini-1.5-flash")
#     model = genai.GenerativeModel("gemini-1.5-turbo")

#     response = model.generate_content(prompt)
#     return response.text

# if __name__ == "__main__":
#     text = "Giảng đường rộng rãi, thiết bị hiện đại, mình rất hài lòng."
#     predicted_sentiment = "Tích cực"
#     predicted_topic = "Cơ sở vật chất"

#     result = check_and_explain_gemini(text, predicted_sentiment, predicted_topic)
#     print(result)


from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("⚠️ Không tìm thấy GEMINI_API_KEY trong file .env")

client = genai.Client(api_key=api_key)

def check_and_explain_gemini(text, predicted_sentiment, predicted_topic):
    prompt = f"""
        Bạn là chuyên gia phân tích văn bản tiếng Việt.
        Dưới đây là một bình luận và kết quả dự đoán từ mô hình:
        Bình luận: "{text}"
        Dự đoán cảm xúc: {predicted_sentiment}
        Dự đoán chủ đề: {predicted_topic}

        Yêu cầu:
        1. Kiểm tra xem dự đoán cảm xúc và chủ đề có hợp lý không.
        2. Nếu thấy sai, hãy đưa ra dự đoán đúng hơn.
        3. Giải thích ngắn gọn tại sao văn bản này lại có cảm xúc/chủ đề đó. Liệt kê vài từ có liên quan đến cảm xúc đó.
        Trả lời dưới dạng JSON với các trường:
        - sentiment
        - topic
        - explanation
        """

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    return response.text

if __name__ == "__main__":
    text = "Giảng đường rộng rãi, thiết bị hiện đại, mình rất hài lòng."
    predicted_sentiment = "Tích cực"
    predicted_topic = "Cơ sở vật chất"

    result = check_and_explain_gemini(text, predicted_sentiment, predicted_topic)
    print(result)
