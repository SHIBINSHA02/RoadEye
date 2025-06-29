# app.py
import streamlit as st
from PIL import Image
import torch
from pathlib import Path
import os

# Load model
model_path = "runs/train/helmet_model11/weights/best.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

st.title("Helmet Detection App")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run detection
    results = model(image)
    results.render()  # updates results.imgs with boxes and labels

    # Display result
    st.image(results.ims[0], caption="Detected Image", use_column_width=True)

