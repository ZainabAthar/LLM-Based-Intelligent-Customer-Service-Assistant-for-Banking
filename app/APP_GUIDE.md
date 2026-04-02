# 🏦 NUST Bank AI Terminal - Setup Guide

This guide explains how to connect your **Streamlit Frontend** (Local) to your **Gwen-4B "Brain"** (Google Colab).

## Prerequisites
1.  **ngrok Account**: Sign up for a free account at [ngrok.com](https://dashboard.ngrok.com).
2.  **Authtoken**: Copy your `Authtoken` from the ngrok dashboard.

---

## Part A: Google Colab (The Brain)
1.  **Ensure Fine-tuning is finished**: Make sure `finetune.py` has saved the weights.
2.  **Move the API file**: Make sure `app/api_service.py` is in your Drive folder.
3.  **Run these commands in a Colab Cell**:
    ```bash
    # Install API dependencies
    !pip install fastapi uvicorn pyngrok
    
    # Start the server (Replace YOUR_TOKEN)
    !python api_service.py --token YOUR_NGROK_AUTHTOKEN_HERE
    ```
4.  **Copy the Link**: Look for the log line: `🔗 Tunnel URL: https://xxxx.ngrok-free.app`. **Copy this link.**

---

## Part B: Your Computer (The Face)
1.  **Open Terminal** in the `colab_deploy/app` folder.
2.  **Install dependencies**:
    ```bash
    pip install streamlit requests
    ```
3.  **Launch the App**:
    ```bash
    streamlit run streamlit_app.py
    ```
4.  **Connect**:
    - Open the URL in your browser (usually `http://localhost:8501`).
    - Paste the **ngrok URL** from Colab into the sidebar.
    - Start Chatting!

---

## Important Tips
- **Timeout**: The first question might take 5-10 seconds while the model loads in Colab.
- **Safety**: Do not share your ngrok URL with others, as it allows anyone to query your GPU session.
- **Session Reset**: If you restart Colab, you will get a NEW ngrok URL. You will need to update it in the Streamlit Sidebar.
