import os
from io import BytesIO

import google.generativeai as genai
import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

st.set_page_config(page_title="Mini Plantix", layout="wide")

load_dotenv()


def get_secret(name):
    value = st.secrets.get(name)
    if value:
        return value
    return os.getenv(name, "")


GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
WEATHER_API_KEY = get_secret("WEATHER_API_KEY")
PLANTNET_API_KEY = get_secret("PLANTNET_API_KEY")
PLANTNET_BASE_URL = "https://my-api.plantnet.org/v2"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def get_weather(city):
    try:
        if not WEATHER_API_KEY:
            raise ValueError("WEATHER_API_KEY is missing")

        url = (
            "https://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={WEATHER_API_KEY}&units=metric"
        )
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data["main"]["temp"], data["main"]["humidity"]
    except Exception:
        return 25, 60


def prepare_image_bytes(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    preview = image.copy()
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return preview, buffer.getvalue()


def call_plantnet(endpoint, image_bytes, organs=None):
    if not PLANTNET_API_KEY:
        raise ValueError("PLANTNET_API_KEY is missing. Add it to your .env file.")

    url = f"{PLANTNET_BASE_URL}/{endpoint}?api-key={PLANTNET_API_KEY}&lang=en"
    files = {"images": ("leaf.jpg", image_bytes, "image/jpeg")}
    data = {}
    if organs:
        data["organs"] = organs

    response = requests.post(url, files=files, data=data, timeout=30)
    response.raise_for_status()
    return response.json()


def identify_plant(image_bytes):
    data = call_plantnet("identify/all", image_bytes, organs="leaf")
    results = data.get("results", [])
    if not results:
        return None

    best = results[0]
    species = best.get("species", {})
    common_names = species.get("commonNames") or []
    return {
        "scientific_name": species.get("scientificNameWithoutAuthor", "Unknown plant"),
        "common_name": common_names[0] if common_names else "Unknown",
        "confidence": best.get("score", 0.0),
        "raw": best,
    }


def identify_disease(image_bytes):
    try:
        data = call_plantnet("health_assessment", image_bytes)
    except requests.HTTPError:
        return None

    results = data.get("results", [])
    if not results:
        return None

    best = results[0]
    return {
        "name": best.get("disease", {}).get("name") or best.get("name") or "Unknown issue",
        "confidence": best.get("score", 0.0),
        "raw": best,
    }


def build_fallback_report(plant_result, disease_result, city, temp, humidity):
    lines = []
    if plant_result:
        lines.append(f"**Plant:** {plant_result['common_name']} ({plant_result['scientific_name']})")
        lines.append(f"**Plant confidence:** {plant_result['confidence']:.2%}")
    else:
        lines.append("**Plant:** Not confidently identified")

    if disease_result and disease_result["confidence"] >= 0.30:
        lines.append(f"**Likely issue:** {disease_result['name']}")
        lines.append(f"**Issue confidence:** {disease_result['confidence']:.2%}")
    else:
        lines.append("**Likely issue:** No confident disease match from API")

    lines.append(f"**Weather in {city}:** {temp} C, {humidity}% humidity")
    lines.append("**Note:** Low confidence may mean the leaf is unclear, healthy, or unsupported by the API.")
    return "\n\n".join(lines)


def analyze_leaf_with_context(image, plant_result, disease_result, city, temp, humidity):
    fallback = build_fallback_report(plant_result, disease_result, city, temp, humidity)
    if not GEMINI_API_KEY:
        return fallback

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""
        You are helping in a mini Plantix style plant health app.

        Use the API results below carefully. If confidence is low, clearly say the diagnosis is uncertain.

        Plant identification:
        - Common name: {plant_result['common_name'] if plant_result else 'Unknown'}
        - Scientific name: {plant_result['scientific_name'] if plant_result else 'Unknown'}
        - Confidence: {plant_result['confidence'] if plant_result else 0}

        Disease identification:
        - Likely issue: {disease_result['name'] if disease_result else 'Unknown'}
        - Confidence: {disease_result['confidence'] if disease_result else 0}

        Weather:
        - City: {city}
        - Temperature: {temp} C
        - Humidity: {humidity}%

        Create a short diagnosis report with these sections:
        1. Plant identified
        2. Likely disease or health status
        3. Confidence and uncertainty note
        4. Likely cause
        5. Treatment in simple English
        6. Prevention tips based on current weather

        Do not claim certainty when confidence is low.
        """
        response = model.generate_content([prompt, image])
        return response.text
    except Exception:
        return fallback


st.title("Mini Plantix - Plant Disease Checker")
st.write(
    "Upload one leaf image to identify the plant, estimate the likely disease, and get weather-aware advice."
)

with st.sidebar:
    st.header("Contact Us")
    st.write("Developed by Yashwant")
    st.markdown("[GitHub](https://github.com/Yashwant2005)")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/yashwant-vashisht-547684261)")
    st.markdown("[Telegram](https://t.me/FLIRTER_KUN)")

col1, col2 = st.columns([1, 1.3])

with col1:
    st.subheader("Image Input")
    image_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])
    city = st.text_input("Enter City Name", "Jaipur")

if image_file and city:
    preview_image, image_bytes = prepare_image_bytes(image_file)
    temp, humidity = get_weather(city)

    with col1:
        st.image(preview_image, caption="Uploaded leaf", use_container_width=True)
        st.info(f"Weather in {city}: {temp} C | Humidity: {humidity}%")

    with col2:
        st.subheader("Diagnosis Report")
        if st.button("Analyze Leaf"):
            with st.spinner("Checking plant identity and likely disease..."):
                try:
                    plant_result = identify_plant(image_bytes)
                    disease_result = identify_disease(image_bytes)
                    report = analyze_leaf_with_context(
                        preview_image,
                        plant_result,
                        disease_result,
                        city,
                        temp,
                        humidity,
                    )

                    if plant_result:
                        st.success(
                            f"Plant identified: {plant_result['common_name']} ({plant_result['scientific_name']})"
                        )
                    else:
                        st.warning("Plant could not be identified confidently.")

                    if disease_result and disease_result["confidence"] >= 0.30:
                        st.info(
                            f"Likely issue: {disease_result['name']} | Confidence: {disease_result['confidence']:.2%}"
                        )
                    else:
                        st.warning("No confident disease match found. The image may be healthy or unsupported.")

                    st.markdown(report)
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")
