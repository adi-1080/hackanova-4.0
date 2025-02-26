import streamlit as st
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
from PIL import Image

# Load pre-trained models and scaler
@st.cache_resource
def load_models():
    # Load the encoder model (replace with your actual model path)
    encoder = tf.keras.models.load_model("encoder.h5")
    # Load the discriminator model (replace with your actual model path)
    discriminator = tf.keras.models.load_model("discriminator.h5")
    # Load the scaler (replace with your actual scaler path)
    scaler = StandardScaler()
    scaler.mean_ = np.load("scaler_mean.npy")
    scaler.scale_ = np.load("scaler_scale.npy")
    return encoder, discriminator, scaler

encoder, discriminator, scaler = load_models()

# Function to preprocess an image
def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (128, 128))  # Resize to 128x128
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
    image = (image / 255.0).astype(np.float32)  # Normalize to [0, 1]
    return image

# Function to extract frames from a video
def extract_frames(video_path, num_frames=10):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (128, 128))  # Resize to 128x128
            frame = (frame / 255.0).astype(np.float32)  # Normalize to [0, 1]
            frames.append(frame)
    cap.release()
    return frames

# Function to detect and crop faces
def detect_and_crop_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image_uint8 = (image * 255).astype(np.uint8)
    gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        cropped_face = image[y:y+h, x:x+w]
    else:
        cropped_face = image  # Return the original image if no face is detected
    resized_face = cv2.resize(cropped_face, (128, 128))
    return resized_face

# Function to predict using the encoder and discriminator
def predict(image):
    # Preprocess the image
    image = preprocess_image(image)
    # Detect and crop face
    cropped_face = detect_and_crop_face(image)
    # Extract latent features
    latent_features = encoder.predict(np.expand_dims(cropped_face, axis=0))[2]
    # Scale the features
    latent_features_scaled = scaler.transform(latent_features)
    # Predict using the discriminator
    prediction = discriminator.predict(latent_features_scaled)
    return prediction[0][0]

# Streamlit app
st.title("Deepfake Detection App")
st.write("Upload an image or video to detect if it is real or fake.")

# File uploader
uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    if uploaded_file.type.startswith('image'):
        # Process image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Classifying...")
        prediction = predict(image)
        st.write(f"Prediction: {'Fake' if prediction > 0.5 else 'Real'}")
        st.write(f"Confidence: {prediction:.2f}")

    elif uploaded_file.type.startswith('video'):
        # Process video
        st.video(uploaded_file)
        st.write("Extracting frames and classifying...")
        video_path = os.path.join("temp", uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        frames = extract_frames(video_path)
        predictions = [predict(frame) for frame in frames]
        avg_prediction = np.mean(predictions)
        st.write(f"Average Prediction: {'Fake' if avg_prediction > 0.5 else 'Real'}")
        st.write(f"Average Confidence: {avg_prediction:.2f}")
        os.remove(video_path)  # Clean up temporary file