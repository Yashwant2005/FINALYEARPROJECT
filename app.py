import os

import google.generativeai as genai
import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

# --- STEP 1: CONFIG & API KEYS ---
st.set_page_config(page_title="Universal Plant Doctor", layout="wide")

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# --- STEP 2: WEATHER FUNCTION ---
def get_weather(city):
    try:
        if not WEATHER_API_KEY:
            raise ValueError("WEATHER_API_KEY is missing")

        url = (
            "https://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={WEATHER_API_KEY}&units=metric"
        )
        res = requests.get(url, timeout=10).json()
        temp = res["main"]["temp"]
        humidity = res["main"]["humidity"]
        return temp, humidity
    except Exception:
        return 25, 60


# --- STEP 3: AI ANALYSIS FUNCTION ---
def analyze_leaf(image, city, temp, humidity):
    try:
        if not GEMINI_API_KEY:
            return "Error: GEMINI_API_KEY is missing. Add it to your .env file."

        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""
        Acting as an expert agricultural pathologist, analyze this plant leaf image.

        Context Info:
        - Location: {city}
        - Current Temperature: {temp} C
        - Current Humidity: {humidity}%

        Provide the following details clearly:
        1. Plant Name
        2. Disease/Health Status
        3. Severity Level (Low/Medium/High)
        4. Scientific Reason (Why this happened?)
        5. Detailed Cure/Treatment (Provide in both Hindi and English)
        6. Prevention Tips based on the current weather in {city}.
        """

        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"


# --- STEP 4: STREAMLIT UI ---
st.title("Universal Plant Disease Detector")
st.write("Upload any plant leaf image from anywhere in the world for AI-based analysis.")

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("Image Input")
    image_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])
    city = st.text_input("Enter City Name for Local Weather", "Jaipur")

if image_file and city:
    image = Image.open(image_file)
    temp, humidity = get_weather(city)

    with col1:
        st.image(image, caption="Uploaded Leaf", use_container_width=True)
        st.info(f"Weather in {city}: {temp} C | Humidity: {humidity}%")

    with col2:
        st.subheader("AI Diagnosis Report")
        if st.button("Analyze Leaf with Global AI"):
            with st.spinner("AI is examining the leaf patterns and cross-referencing global data..."):
                report = analyze_leaf(image, city, temp, humidity)
                st.markdown(report)
                st.success("Analysis complete")
