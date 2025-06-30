# Automated-Face-Hand-Segmentation-Using-SAM2-API
This allows users to upload an image and instantly obtain segmented masks for faces and hands, all within a user-friendly Streamlit Interface.

# Features 
Upload any image and get segmented face and hand.

Uses SAM2 via Replicate API.

Includes YOLOv8n for object detection.

Streamlit based UI interface for user friendly interface.


# Activate virtual Environment (Optional)
step 1- python -m venv venv
step 2- venv\Scripts\activate


# Requirements.txt
This file include required libraries to implement this project smoothly.
In terminal perform :-  pip install -r requirements.txt


# Step up your .env file
REPLICATE_API_TOKEN=your_replicate_api_key_here
( Get your free API key from https://replicate.com/account)


# Key points 
Use latest Replicate version for better result and smooth functioning.


# Run the Project
streamlit run app.py
