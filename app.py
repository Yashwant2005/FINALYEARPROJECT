import os
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageOps

st.set_page_config(page_title="GreenLeaf", layout="wide")

load_dotenv()


def get_secret(name):
    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass
    return os.getenv(name, "")


GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
WEATHER_API_KEY = get_secret("WEATHER_API_KEY")
PLANTNET_API_KEY = get_secret("PLANTNET_API_KEY")
PLANTNET_BASE_URL = "https://my-api.plantnet.org/v2"

BACKGROUND_IMAGE_URL = (
    "https://static.vecteezy.com/system/resources/thumbnails/026/937/720/"
    "small_2x/ai-generative-beautiful-dark-green-nature-background-fern-leaves-black-green-background-for-design-web-banner-website-header-exotic-plants-closeup-photo.jpg"
)

st.markdown(
    f"""
    <style>
        .stApp {{
            background:
                linear-gradient(rgba(6, 24, 14, 0.72), rgba(6, 24, 14, 0.72)),
                url("{BACKGROUND_IMAGE_URL}") center center / cover fixed no-repeat;
        }}
        [data-testid="stHeader"] {{ background: rgba(0, 0, 0, 0); }}
        [data-testid="stSidebar"] {{ background: rgba(7, 24, 15, 0.88); }}
        [data-testid="stAppViewContainer"] {{ color: #f2f7f2; }}
        [data-testid="stFileUploader"],
        [data-testid="stTextInput"],
        [data-testid="stForm"],
        .stAlert,
        .stButton > button {{ border-radius: 12px; }}
        .block-container {{
            max-width: 1200px;
            padding-top: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }}
        h1, h2, h3, p, label, div {{ color: #f2f7f2; }}
        @media (max-width: 900px) {{
            .stApp {{ background-attachment: scroll; }}
            .block-container {{ padding-left: 1rem; padding-right: 1rem; padding-top: 1.1rem; }}
            h1 {{ font-size: 2.2rem !important; }}
            h2 {{ font-size: 1.7rem !important; }}
            [data-testid="stHorizontalBlock"] {{ flex-direction: column !important; gap: 0.8rem !important; }}
            [data-testid="stHorizontalBlock"] > div {{ min-width: 100% !important; width: 100% !important; }}
            .stButton > button {{ width: 100%; }}
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=600, show_spinner=False)
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
    """
    Robust image loader that handles:
    - WebP, BMP, TIFF (common when saving from Google/browser)
    - EXIF rotation (phone/Google photos)
    - RGBA, palette, CMYK → converts all to RGB
    - Re-encodes as JPEG so PlantNet API always accepts it
    """
    raw_bytes = uploaded_file.read()
    buf = BytesIO(raw_bytes)

    try:
        image = Image.open(buf)
        image.load()  # force decode so corrupt files fail here, not later
    except Exception as e:
        raise ValueError(
            f"Could not open image: {e}. "
            "Try right-clicking the image → Save as → JPEG/PNG, then upload that file."
        )

    # Fix EXIF rotation (Google photos, phone cameras often have this)
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    # For animated formats (WebP/GIF), take first frame only
    try:
        if getattr(image, "n_frames", 1) > 1:
            image.seek(0)
    except Exception:
        pass

    # Normalise colour mode to RGB
    if image.mode == "P":
        image = image.convert("RGBA")
    if image.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", image.size, (255, 255, 255))
        alpha = image.split()[-1]
        bg.paste(image.convert("RGB"), mask=alpha)
        image = bg
    elif image.mode != "RGB":
        image = image.convert("RGB")

    preview = image.copy()

    # Always re-encode as JPEG — PlantNet rejects WebP/BMP/TIFF
    out = BytesIO()
    image.save(out, format="JPEG", quality=92)
    return preview, out.getvalue()


def call_plantnet(endpoint, image_bytes, field="images", organs=None):
    """Generic PlantNet POST. Use field='images' for identify, field='image' for diseases."""
    if not PLANTNET_API_KEY:
        raise ValueError("PLANTNET_API_KEY is missing. Add it to your .env file.")

    url = f"{PLANTNET_BASE_URL}/{endpoint}?api-key={PLANTNET_API_KEY}&lang=en"
    files = {field: ("leaf.jpg", image_bytes, "image/jpeg")}
    data = {}
    if organs:
        data["organs"] = organs

    last_exc = None
    for attempt in range(3):
        try:
            response = requests.post(url, files=files, data=data, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(0.6 * (attempt + 1))
    raise last_exc


def identify_plant(image_bytes):
    # Identify endpoint uses field name "images" (plural)
    data = call_plantnet("identify/all", image_bytes, field="images", organs="leaf")
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
    # Disease endpoint uses field name "image" (singular) — confirmed by PlantNet curl docs
    # Endpoint: POST /v2/diseases/identify
    try:
        data = call_plantnet("diseases/identify", image_bytes, field="image")
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        try:
            msg = exc.response.json().get("message", str(exc))
        except Exception:
            msg = str(exc)
        raise RuntimeError(f"Disease API error {status}: {msg}") from exc
    except requests.RequestException:
        return None

    results = data.get("results", [])
    if not results:
        return None

    best = results[0]
    disease_info = best.get("disease", {})
    name = (
        disease_info.get("name")
        or disease_info.get("scientificName")
        or best.get("name")
        or "Unknown issue"
    )
    return {
        "name": name,
        "confidence": best.get("score", 0.0),
        "raw": best,
    }


def safe_identify_plant(image_bytes):
    try:
        return identify_plant(image_bytes), None
    except requests.RequestException:
        return None, "Plant identification service is temporarily unavailable."
    except Exception:
        return None, "Plant identification could not be completed for this image."


def safe_identify_disease(image_bytes):
    try:
        return identify_disease(image_bytes), None
    except RuntimeError as exc:
        # Surface real API errors so they are visible in the UI
        return None, str(exc)
    except requests.RequestException:
        return None, "Disease analysis service is temporarily unavailable."
    except Exception as exc:
        return None, f"Disease analysis failed: {exc}"


def detect_visible_leaf_stress(image):
    sample = image.copy()
    sample.thumbnail((256, 256))

    green_pixels = 0
    lesion_pixels = 0

    for r, g, b in sample.getdata():
        if g > 55 and g >= r * 0.85 and g >= b * 1.05:
            green_pixels += 1
        elif (
            (r > 110 and g > 45 and r >= g * 1.08 and b < g * 0.95)
            or (r > 85 and 35 < g < 150 and b < 110 and r >= g * 1.02)
        ):
            lesion_pixels += 1

    total_leaf_like = green_pixels + lesion_pixels
    if total_leaf_like == 0:
        return {
            "visible_stress": False,
            "severity": "unknown",
            "ratio": 0.0,
            "summary": "Visible symptoms could not be estimated from the image.",
        }

    lesion_ratio = lesion_pixels / total_leaf_like
    if lesion_ratio >= 0.12:
        severity = "high"
    elif lesion_ratio >= 0.05:
        severity = "moderate"
    else:
        severity = "low"

    visible_stress = lesion_ratio >= 0.03
    if visible_stress:
        summary = (
            f"Visible spotting and tissue damage are present on the leaf "
            f"({severity} severity estimate)."
        )
    else:
        summary = "No strong visible spotting pattern was detected from the image alone."

    return {
        "visible_stress": visible_stress,
        "severity": severity,
        "ratio": lesion_ratio,
        "summary": summary,
    }


def build_fallback_report(plant_result, disease_result, city, temp, humidity, symptom_check):
    lines = []
    if plant_result:
        lines.append(f"**Plant:** {plant_result['common_name']} ({plant_result['scientific_name']})")
        lines.append(f"**Plant confidence:** {plant_result['confidence']:.2%}")
    else:
        lines.append("**Plant:** Not confidently identified")

    if disease_result and disease_result["confidence"] >= 0.30:
        lines.append(f"**Likely issue:** {disease_result['name']}")
        lines.append(f"**Issue confidence:** {disease_result['confidence']:.2%}")
    elif symptom_check["visible_stress"]:
        lines.append("**Likely issue:** The leaf appears diseased, but the exact condition is uncertain")
        lines.append(f"**Visible symptoms:** {symptom_check['summary']}")
    else:
        lines.append("**Likely issue:** No exact disease match was found")

    lines.append(f"**Weather in {city}:** {temp} C, {humidity}% humidity")
    lines.append("**Note:** A low-confidence result means the exact disease name is uncertain, not that the leaf is healthy.")
    return "\n\n".join(lines)


@st.cache_resource(show_spinner=False)
def get_gemini_model():
    if not GEMINI_API_KEY:
        return None, "missing"
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        return client, "google_genai"
    except Exception:
        return None, "unavailable"


def analyze_leaf_with_context(image, plant_result, disease_result, city, temp, humidity, symptom_check):
    fallback = build_fallback_report(plant_result, disease_result, city, temp, humidity, symptom_check)
    model, provider = get_gemini_model()
    if not model:
        return fallback

    try:
        prompt = f"""
        You are helping in the GreenLeaf plant health app.

        Use the plant and disease lookup results carefully. If confidence is low, clearly say the diagnosis is uncertain.

        Plant identification:
        - Common name: {plant_result['common_name'] if plant_result else 'Unknown'}
        - Scientific name: {plant_result['scientific_name'] if plant_result else 'Unknown'}
        - Confidence: {plant_result['confidence'] if plant_result else 0}

        Disease identification:
        - Likely issue: {disease_result['name'] if disease_result else 'Unknown'}
        - Confidence: {disease_result['confidence'] if disease_result else 0}

        Visible symptom check from the image:
        - Visible stress detected: {symptom_check['visible_stress']}
        - Severity estimate: {symptom_check['severity']}
        - Summary: {symptom_check['summary']}

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

        If the leaf visibly looks diseased, say that clearly even when the exact disease name is uncertain.
        Do not claim certainty when confidence is low.
        """

        if provider == "google_genai":
            response = model.models.generate_content(
                model="gemini-1.5-flash",
                contents=[prompt, image],
            )
            return response.text or fallback

        response = model.generate_content([prompt, image])
        return response.text
    except Exception:
        return fallback


# ── UI ──────────────────────────────────────────────────────────────────────

st.title("GreenLeaf")
st.write(
    "Upload one leaf image to identify the plant, estimate the likely disease, and get weather-aware advice."
)

with st.sidebar:
    st.header("Contact Us")
    st.write("Developed by Yashwant Vashisht")
    st.markdown("[GitHub](https://github.com/Yashwant2005)")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/yashwant-vashisht-547684261)")
    st.markdown("[Telegram](https://t.me/FLIRTER_KUN)")

col1, col2 = st.columns([1, 1.3])

with col1:
    st.subheader("Image Input")
    image_file = st.file_uploader(
        "Upload Leaf Image",
        # Accept everything browsers typically save images as
        type=["jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif"],
        help="Tip: if a Google/browser image fails, right-click → Save image as → JPEG, then upload.",
    )
    city = st.text_input("Enter City Name", "Jaipur")

# Process image outside columns so errors show in col1
preview_image = None
image_bytes = None
load_error = None

if image_file:
    try:
        preview_image, image_bytes = prepare_image_bytes(image_file)
    except ValueError as e:
        load_error = str(e)

with col1:
    if load_error:
        st.error(f"⚠️ {load_error}")
    elif preview_image:
        st.image(preview_image, caption="Uploaded leaf", width="stretch")
        temp, humidity = get_weather(city)
        st.info(f"Weather in {city}: {temp} °C | Humidity: {humidity}%")

if preview_image and city:
    symptom_check = detect_visible_leaf_stress(preview_image)

    with col2:
        st.subheader("Diagnosis Report")
        if st.button("Analyze Leaf"):
            with st.spinner("Analyzing leaf health..."):
                try:
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        plant_future = executor.submit(safe_identify_plant, image_bytes)
                        disease_future = executor.submit(safe_identify_disease, image_bytes)
                        plant_result, plant_error = plant_future.result()
                        disease_result, disease_error = disease_future.result()

                    report = analyze_leaf_with_context(
                        preview_image,
                        plant_result,
                        disease_result,
                        city,
                        temp,
                        humidity,
                        symptom_check,
                    )

                    if plant_result:
                        st.success(
                            f"Plant identified: {plant_result['common_name']} ({plant_result['scientific_name']})"
                        )
                    else:
                        st.warning("Plant could not be identified confidently.")
                    if plant_error:
                        st.info(plant_error)

                    if disease_result and disease_result["confidence"] >= 0.30:
                        st.info(
                            f"Likely issue: {disease_result['name']} | Confidence: {disease_result['confidence']:.2%}"
                        )
                    elif symptom_check["visible_stress"]:
                        st.warning(
                            "Visible disease symptoms were detected, but the exact condition could not be named confidently."
                        )
                    else:
                        st.warning("No exact disease match was found. Try a closer and sharper leaf photo.")
                    if disease_error:
                        st.info(disease_error)

                    st.markdown(report)
                except Exception:
                    st.error("Analysis could not be completed right now. Please retry in a few seconds.")