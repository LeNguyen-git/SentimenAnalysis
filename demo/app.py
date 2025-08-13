
import sys
import os
sys.path.append('../bert')

from bert_predict import predict_sentiment, predict_topic, init_models
from bert_predict_file import (
    predict_file,
    summarize_sentiment_reasons_by_topic,
    class_names,
    STOPWORDS,
    explainer,
    lime_predict_proba
)
from pyvi import ViTokenizer
from bert_api_explained import check_and_explain_gemini


from fastapi import FastAPI, Request, File, UploadFile, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import uvicorn


app = FastAPI()
templates = Jinja2Templates(directory="templates")

model_label, model_topic, tokenizer, preprocessor, device = init_models()
last_results = None
last_stats = None


class TextInput(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/bert-predict")
async def predict_text_api(text_input: TextInput):
    text = text_input.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Vui lòng nhập văn bản")

    result_sentiment = predict_sentiment(model_label, tokenizer, preprocessor, text, device)
    result_topic = predict_topic(model_topic, tokenizer, preprocessor, text, device)

    return {
        "sentiment": result_sentiment,
        "topic": result_topic
    }


@app.post("/bert-predict-file")
async def predict_file_lime(request: Request, file: UploadFile = File(...)):
    global last_results, last_stats

    form_data = await request.form()
    print(f"Raw form data: {dict(form_data)}")

    save_value = form_data.get("save", "false")
    print(f"Received save parameter: {save_value}")

    if not file.filename:
        raise HTTPException(status_code=400, detail="File không hợp lệ")

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    save_bool = save_value.lower() == "true"
    print(f"save_bool: {save_bool}")

    output_path = None
    if save_bool:
        output_path = os.path.join(upload_dir, f"result_{file.filename}")
        if not output_path.lower().endswith(".csv"):
            base_name = os.path.splitext(output_path)[0]
            output_path = f"{base_name}.csv"

    results, stats = predict_file(
        file_path,
        output_path=output_path,
        save=save_bool,
        model_label=model_label,
        model_topic=model_topic,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        device=device
    )

    last_results = results
    last_stats = stats

    download_url = None
    if save_bool and output_path:
        filename = os.path.basename(output_path)
        download_url = f"/uploads/{filename}"

    return {
        "results": results,
        "stats": stats,
        "download_url": download_url
    }


@app.post("/explain-text")
async def explain_single_text(text_input: TextInput):
    text = text_input.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Vui lòng nhập văn bản")

    sentiment = predict_sentiment(model_label, tokenizer, preprocessor, text, device)["label"]
    topic = predict_topic(model_topic, tokenizer, preprocessor, text, device)["topic"]
    label_idx = class_names.index(sentiment)

    exp = explainer.explain_instance(
        ViTokenizer.tokenize(text),
        lime_predict_proba,
        num_features=10,
        labels=[label_idx],
        num_samples=60
    )

    keywords = [
        word
        for word, weight in exp.as_list(label=label_idx)
        if weight > 0 or word.lower() not in STOPWORDS
    ]

    return {
        "text": text,
        "sentiment": sentiment,
        "topic": topic,
        "keywords": keywords
    }

@app.get("/explain")
async def explain_sentiment_api(
    target_sentiment: str = Query(...),
    target_topic: str = Query(...)
):
    if last_results is None:
        raise HTTPException(status_code=400, detail="Chưa có kết quả dự đoán để giải thích")

    if target_sentiment not in class_names:
        raise HTTPException(status_code=400, detail=f"Cảm xúc không hợp lệ, phải thuộc {class_names}")

    explanations = summarize_sentiment_reasons_by_topic(
        last_results,
        target_sentiment=target_sentiment,
        target_topic=target_topic
    )

    return {
        "target_sentiment": target_sentiment,
        "target_topic": target_topic,
        "explanations": explanations
    }

@app.post("/explain-text-api")
async def explain_single_text_api(text_input: TextInput):
    text = text_input.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Vui lòng nhập văn bản")

    sentiment_result = predict_sentiment(model_label, tokenizer, preprocessor, text, device)
    topic_result = predict_topic(model_topic, tokenizer, preprocessor, text, device)

    sentiment = sentiment_result["label"]
    topic = topic_result["topic"]

    explanation = check_and_explain_gemini(
        text,
        sentiment,
        topic
    )

    return {
        "textapi": text,
        "sentimentapi": sentiment,
        "topicapi": topic,
        "explanationsapi": explanation
    }

@app.get("/explain-file-api")
async def explain_file_api(
    target_topic: str = Query(..., description="Chủ đề cần giải thích"),
    target_sentiment: str = Query(..., description="Cảm xúc cần giải thích")
):
    global last_results

    if not last_results:
        raise HTTPException(status_code=400, detail="Chưa có kết quả dự đoán để giải thích")

    texts = [
        r["text"] for r in last_results
        if r["predicted_sentiment"] == target_sentiment and r["predicted_topic"] == target_topic
    ]

    if not texts:
        raise HTTPException(status_code=404, detail="Không tìm thấy bình luận phù hợp")

    combined_text = "\n---\n".join(texts)

    explanation = check_and_explain_gemini(combined_text, target_sentiment, target_topic)

    return {
        "top_topic": target_topic,
        "top_sentiment": target_sentiment,
        "total_texts": texts,
        "exp_result": explanation
    }



if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=5000)
