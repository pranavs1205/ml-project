from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
import io
import sys
import os

# Add src to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.inference import SentimentPredictor, InferenceConfig, add_sentiment_prediction

app = FastAPI(
    title="Sentiment Analysis API",
    description="API to predict sentiment from movie reviews.",
    version="1.0.0"
)

predictor = None

@app.on_event("startup")
async def startup_event():
    global predictor
    try:
        infer_config = InferenceConfig(model_path="models/sentiment_cnn.pth")
        predictor = SentimentPredictor(config=infer_config)
        print("✅ Model loaded successfully for API.")
    except FileNotFoundError:
        print("⚠️ Model file not found. Please train the model first.")
        predictor = None

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    text: str
    sentiment: str

@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "API is running"}

@app.post("/predict_text", response_model=SentimentResponse, tags=["Prediction"])
async def predict_sentiment_text(request: SentimentRequest):
    """Predicts sentiment for a single review text."""
    if not predictor:
        raise HTTPException(status_code=503, detail="Model is not available.")
    
    prediction = predictor.predict(request.text)
    return SentimentResponse(text=request.text, sentiment=prediction)

@app.post("/predict_csv", tags=["Prediction"])
async def predict_sentiment_csv(file: UploadFile = File(...)):
    """Processes a CSV file with a 'review' column and returns predictions."""
    if not predictor:
        raise HTTPException(status_code=503, detail="Model is not available.")
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV.")

    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    if 'review' not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain a 'review' column.")
        
    results_df = add_sentiment_prediction(df, predictor)
    return results_df.to_dict(orient='records')