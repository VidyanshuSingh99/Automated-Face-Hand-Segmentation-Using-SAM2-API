# app.py
import streamlit as st
import os
from detect import detect_objects
from segment import get_mask_sam2
import cv2
from PIL import Image
import numpy as np

st.title("🧠 Automated Face & Hand Segmentation using SAM2")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])
if not os.path.exists("uploads"):
    os.makedirs("uploads")

if uploaded_file:
    image_path = f"uploads/{uploaded_file.name}"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.read())

    st.image(image_path, caption="Uploaded Image", use_column_width=True)

    image, boxes = detect_objects(image_path)
    st.success(f"Detected {len(boxes)} objects")

    sam_output = get_mask_sam2(image_path, boxes)
    st.json(sam_output)  
