from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import librosa
import tensorflow as tf
from tempfile import NamedTemporaryFile
import os
from typing import Optional
import uvicorn
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="Audio Deepfake Detection API",
    description="API for detecting AI-generated deepfake audio using deep learning",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model = tf.keras.models.load_model("deepfake_audio_model.h5")

# Response model
class PredictionResult(BaseModel):
    prediction: str
    confidence: float
    is_fake: bool
    probabilities: dict

# Feature extraction function
def extract_advanced_features(file_path: str, max_time_steps: int = 130) -> np.ndarray:
    """
    Extract features from an audio file using the same method as training.
    """
    try:
        y, sr = librosa.load(file_path, sr=22050, duration=3.0)

        # Compute Mel-spectrogram
        mels = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=128, fmax=8000, 
            n_fft=2048, hop_length=512
        )
        log_mels = librosa.power_to_db(mels, ref=np.max)

        # Enforce fixed time dimension
        if log_mels.shape[1] > max_time_steps:
            log_mels = log_mels[:, :max_time_steps]
        elif log_mels.shape[1] < max_time_steps:
            pad_width = max_time_steps - log_mels.shape[1]
            log_mels = np.pad(log_mels, ((0, 0), (0, pad_width)), mode='constant')

        # Compute deltas
        delta = librosa.feature.delta(log_mels)
        delta2 = librosa.feature.delta(log_mels, order=2)

        # Stack features and normalize
        features = np.stack([log_mels, delta, delta2], axis=-1)
        features = (features - np.mean(features)) / (np.std(features) + 1e-9)

        return features
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio processing error: {str(e)}")

@app.post("/predict", response_model=PredictionResult)
async def predict_audio(file: UploadFile = File(...)):
    """
    Predict whether an uploaded audio file is real or fake (deepfake).
    
    Accepts WAV or MP3 files (3-5 seconds works best).
    """
    # Validate file type
    if not file.filename.lower().endswith(('.wav', '.mp3')):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload a WAV or MP3 file."
        )

    try:
        # Save the uploaded file temporarily
        with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # Extract features
        features = extract_advanced_features(tmp_file_path)
        features = np.expand_dims(features, axis=0)

        # Make prediction
        y_pred_prob = model.predict(features).ravel()[0]
        y_pred = 1 if y_pred_prob > 0.5 else 0
        confidence = max(y_pred_prob, 1 - y_pred_prob)

        # Prepare response
        result = {
            "prediction": "fake" if y_pred == 1 else "real",
            "confidence": float(confidence),
            "is_fake": bool(y_pred == 1),
            "probabilities": {
                "real": float(1 - y_pred_prob),
                "fake": float(y_pred_prob)
            }
        }

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temporary file
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

@app.get("/")
async def root():
    return {
        "message": "Audio Deepfake Detection API",
        "usage": "POST /predict with an audio file to get predictions",
        "note": "Works best with 3-5 second WAV or MP3 files"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)