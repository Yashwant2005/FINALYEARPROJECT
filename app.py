import os
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError

st.set_page_config(
    page_title="GreenLeaf – Plant Health AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

# ─── Styles ──────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(135deg, #071a0e 0%, #0d2b18 60%, #0a2010 100%);
            font-family: 'Segoe UI', system-ui, sans-serif;
        }
        [data-testid="stHeader"]   { background: transparent; }
        [data-testid="stSidebar"]  { background: rgba(5, 18, 10, 0.95); border-right: 1px solid rgba(74,222,128,0.15); }
        [data-testid="stAppViewContainer"] { color: #e8f5e9; }
        h1, h2, h3, h4, p, label, div, span { color: #e8f5e9 !important; }
        .block-container { max-width: 1240px; padding: 1.8rem 2rem 3rem; }

        .hero-banner {
            background: linear-gradient(120deg, rgba(22,101,52,0.55) 0%, rgba(20,83,45,0.35) 100%);
            border: 1px solid rgba(74,222,128,0.25);
            border-radius: 20px;
            padding: 2rem 2.5rem 1.6rem;
            margin-bottom: 2rem;
            text-align: center;
        }
        .hero-banner h1 {
            font-size: 2.8rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, #86efac, #4ade80, #22c55e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.4rem !important;
        }
        .hero-banner p { color: #a7f3d0 !important; font-size: 1.05rem; margin: 0; }

        .gl-card {
            background: rgba(20, 60, 35, 0.55);
            border: 1px solid rgba(74,222,128,0.18);
            border-radius: 16px;
            padding: 1.4rem 1.5rem;
            margin-bottom: 1.2rem;
            backdrop-filter: blur(8px);
        }
        .gl-card-title {
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            color: #4ade80 !important;
            margin-bottom: 0.7rem;
        }

        .weather-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(16,185,129,0.15);
            border: 1px solid rgba(16,185,129,0.35);
            border-radius: 50px;
            padding: 0.4rem 1rem;
            font-size: 0.92rem;
            color: #6ee7b7 !important;
            font-weight: 600;
        }

        .conf-bar-wrap {
            background: rgba(255,255,255,0.08);
            border-radius: 20px;
            height: 8px;
            margin-top: 6px;
            overflow: hidden;
        }
        .conf-bar-fill {
            height: 8px;
            border-radius: 20px;
            background: linear-gradient(90deg, #22c55e, #86efac);
        }

        .stButton > button {
            background: linear-gradient(135deg, #16a34a, #15803d) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.6rem 1.6rem !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
            width: 100%;
            box-shadow: 0 4px 15px rgba(22,163,74,0.35) !important;
        }
        .stButton > button:hover { opacity: 0.9 !important; }

        [data-testid="stFileUploader"] {
            background: rgba(20,60,35,0.4) !important;
            border: 2px dashed rgba(74,222,128,0.3) !important;
            border-radius: 14px !important;
        }

        div[data-baseweb="input"] input {
            background: rgba(20,60,35,0.5) !important;
            border: 1px solid rgba(74,222,128,0.3) !important;
            border-radius: 10px !important;
            color: #e8f5e9 !important;
        }

        .steps-row {
            display: flex;
            justify-content: center;
            gap: 0;
            margin: 0.5rem 0 1.8rem;
            flex-wrap: wrap;
        }
        .step-item {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.8rem;
            color: #6ee7b7 !important;
            font-weight: 600;
        }
        .step-dot {
            width: 26px; height: 26px;
            border-radius: 50%;
            background: rgba(34,197,94,0.25);
            border: 2px solid #22c55e;
            display: flex; align-items: center; justify-content: center;
            font-size: 0.75rem; font-weight: 700; color: #86efac !important;
        }
        .step-line { width: 40px; height: 2px; background: rgba(74,222,128,0.3); margin: 0 0.3rem; }

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li { color: #d1fae5 !important; line-height: 1.7; }
        [data-testid="stMarkdownContainer"] strong { color: #86efac !important; }

        .sidebar-link a {
            color: #6ee7b7 !important;
            text-decoration: none;
            font-weight: 600;
            display: block;
            padding: 0.4rem 0;
            border-bottom: 1px solid rgba(74,222,128,0.1);
        }

        @media (max-width: 900px) {
            .hero-banner h1 { font-size: 2rem !important; }
            .block-container { padding: 1rem 0.8rem 2rem; }
            [data-testid="stHorizontalBlock"] { flex-direction: column !important; }
            [data-testid="stHorizontalBlock"] > div { min-width: 100% !important; width: 100% !important; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Helpers ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def get_weather(city):
    try:
        if not WEATHER_API_KEY:
            raise ValueError("WEATHER_API_KEY missing")
        url = (
            "https://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={WEATHER_API_KEY}&units=metric"
        )
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        d = r.json()
        return d["main"]["temp"], d["main"]["humidity"]
    except Exception:
        return 25, 60


def prepare_image_bytes(uploaded_file):
    """
    Robustly convert ANY uploaded image to a clean JPEG buffer.

    Fixes for Google-downloaded images:
    - WebP / AVIF / HEIC formats that PIL normally chokes on
    - Progressive / truncated JPEGs
    - Images with EXIF rotation
    - Very large images that need to be re-encoded
    """
    raw = uploaded_file.read()
    uploaded_file.seek(0)   # rewind so Streamlit can re-display it

    try:
        image = Image.open(BytesIO(raw))
        image.load()        # force full decode — catches truncated files early
    except UnidentifiedImageError:
        raise ValueError(
            "This file could not be recognised as an image. "
            "Please save it as JPEG or PNG and try again."
        )
    except Exception as exc:
        raise ValueError(
            f"Image could not be opened ({exc}). "
            "Try right-clicking the Google image → Save as → JPEG."
        )

    # Strip animation frames, palette mode, alpha channel, EXIF rotation
    if hasattr(image, "n_frames") and image.n_frames > 1:
        image.seek(0)               # take first frame of GIF/APNG
    image = image.convert("RGB")    # drop alpha; normalises all modes

    # Re-encode to a fresh JPEG (removes any corrupt metadata / progressive encoding)
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=90, optimize=True)
    return image, buf.getvalue()


def call_plantnet(endpoint, image_bytes, organs="leaf"):
    """
    Call a PlantNet v2 endpoint.

    PlantNet requires the multipart form to be sent as a list of tuples,
    NOT a dict — otherwise the "images" field is dropped and the API
    returns 400: "Images" is required.

    Both identify and health_assessment also need an `organs` value.
    """
    if not PLANTNET_API_KEY:
        raise ValueError("PLANTNET_API_KEY missing — add it to .env")
    url = f"{PLANTNET_BASE_URL}/{endpoint}?api-key={PLANTNET_API_KEY}&lang=en"

    # Use list-of-tuples so requests sends the correct multipart field names
    files = [("images", ("leaf.jpg", image_bytes, "image/jpeg"))]
    data = [("organs", organs)]          # organs is required by both endpoints

    last_exc = None
    for attempt in range(3):
        try:
            r = requests.post(url, files=files, data=data, timeout=30)
            # Surface the error body for easier debugging
            if not r.ok:
                raise requests.HTTPError(
                    f"{r.status_code}: {r.text[:200]}", response=r
                )
            return r.json()
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(0.6 * (attempt + 1))
    raise last_exc


def identify_plant(image_bytes):
    data = call_plantnet("identify/all", image_bytes, organs="leaf")
    results = data.get("results", [])
    if not results:
        return None
    best = results[0]
    species = best.get("species", {})
    common_names = species.get("commonNames") or []
    return {
        "scientific_name": species.get("scientificNameWithoutAuthor", "Unknown"),
        "common_name": common_names[0] if common_names else "Unknown",
        "confidence": best.get("score", 0.0),
        "raw": best,
    }


def identify_disease(image_bytes):
    try:
        data = call_plantnet("health_assessment", image_bytes, organs="leaf")
    except requests.RequestException:
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
    except requests.HTTPError as exc:
        # Show the actual API error text so it's easier to debug
        return None, f"Disease API error {exc}"
    except requests.RequestException:
        return None, "Disease analysis service is temporarily unavailable."
    except Exception as exc:
        return None, f"Disease analysis error: {exc}"


def detect_visible_leaf_stress(image):
    sample = image.copy()
    sample.thumbnail((256, 256))
    green_pixels = lesion_pixels = 0
    for r, g, b in sample.getdata():
        if g > 55 and g >= r * 0.85 and g >= b * 1.05:
            green_pixels += 1
        elif (
            (r > 110 and g > 45 and r >= g * 1.08 and b < g * 0.95)
            or (r > 85 and 35 < g < 150 and b < 110 and r >= g * 1.02)
        ):
            lesion_pixels += 1
    total = green_pixels + lesion_pixels
    if total == 0:
        return {"visible_stress": False, "severity": "unknown", "ratio": 0.0,
                "summary": "Visible symptoms could not be estimated from the image."}
    ratio = lesion_pixels / total
    severity = "high" if ratio >= 0.12 else "moderate" if ratio >= 0.05 else "low"
    visible_stress = ratio >= 0.03
    summary = (
        f"Visible spotting / tissue damage detected ({severity} severity estimate)."
        if visible_stress
        else "No strong visible spotting pattern detected."
    )
    return {"visible_stress": visible_stress, "severity": severity,
            "ratio": ratio, "summary": summary}


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
        lines.append("**Likely issue:** The leaf appears diseased, but exact condition is uncertain")
        lines.append(f"**Visible symptoms:** {symptom_check['summary']}")
    else:
        lines.append("**Likely issue:** No exact disease match found")
    lines.append(f"**Weather – {city}:** {temp} °C | Humidity {humidity}%")
    lines.append("**Note:** Low confidence means uncertain diagnosis, not a healthy leaf.")
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
        Use the plant/disease lookup results carefully. If confidence is low, say so clearly.

        Plant identification:
        - Common name: {plant_result['common_name'] if plant_result else 'Unknown'}
        - Scientific name: {plant_result['scientific_name'] if plant_result else 'Unknown'}
        - Confidence: {plant_result['confidence'] if plant_result else 0}

        Disease identification:
        - Likely issue: {disease_result['name'] if disease_result else 'Unknown'}
        - Confidence: {disease_result['confidence'] if disease_result else 0}

        Visible symptom check:
        - Stress detected: {symptom_check['visible_stress']}
        - Severity: {symptom_check['severity']}
        - Summary: {symptom_check['summary']}

        Weather: {city} | {temp} C | {humidity}% humidity

        Write a clear diagnosis report with these sections:
        1. Plant identified
        2. Likely disease or health status
        3. Confidence & uncertainty note
        4. Likely cause
        5. Treatment (simple English)
        6. Prevention tips based on current weather

        If the leaf visibly looks diseased, say so even when the exact disease is uncertain.
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


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding: 1rem 0 0.5rem;'>
            <span style='font-size:2.5rem;'>🌿</span>
            <div style='font-size:1.2rem; font-weight:800; color:#86efac !important;
                        letter-spacing:-0.3px; margin-top:0.3rem;'>GreenLeaf</div>
            <div style='font-size:0.78rem; color:#6ee7b7 !important;'>Plant Health AI</div>
        </div>
        <hr style='border-color:rgba(74,222,128,0.2); margin: 0.8rem 0;'>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("##### 🔍 How it works")
    st.markdown(
        """
        <ol style='color:#a7f3d0 !important; font-size:0.88rem; line-height:2;'>
            <li>Upload a clear leaf photo</li>
            <li>Enter your city</li>
            <li>Click <strong style='color:#4ade80!important;'>Analyze Leaf</strong></li>
            <li>Get AI-powered diagnosis</li>
        </ol>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <hr style='border-color:rgba(74,222,128,0.2);'>
        <div style='font-size:0.78rem; color:#6ee7b7 !important; font-weight:700;
                    text-transform:uppercase; letter-spacing:1px; margin-bottom:0.6rem;'>
            💡 Tips for best results
        </div>
        <ul style='color:#a7f3d0 !important; font-size:0.83rem; line-height:1.9;'>
            <li>Use natural daylight</li>
            <li>Frame a single leaf clearly</li>
            <li>Avoid blurry or dark images</li>
            <li>Google images: right-click → Save as JPEG</li>
        </ul>
        <hr style='border-color:rgba(74,222,128,0.2);'>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("##### 👨‍💻 Developer")
    st.markdown(
        """
        <div class='sidebar-link'>
            <a href='https://github.com/Yashwant2005' target='_blank'>🐙 GitHub – Yashwant2005</a>
            <a href='https://www.linkedin.com/in/yashwant-vashisht-547684261' target='_blank'>💼 LinkedIn</a>
            <a href='https://t.me/FLIRTER_KUN' target='_blank'>✈️ Telegram</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class='hero-banner'>
        <h1>🌿 GreenLeaf</h1>
        <p>AI-powered plant disease detection — upload a leaf, get an instant diagnosis</p>
    </div>
    <div class='steps-row'>
        <div class='step-item'><div class='step-dot'>1</div>&nbsp;Upload Image</div>
        <div class='step-line'></div>
        <div class='step-item'><div class='step-dot'>2</div>&nbsp;Enter City</div>
        <div class='step-line'></div>
        <div class='step-item'><div class='step-dot'>3</div>&nbsp;Analyze</div>
        <div class='step-line'></div>
        <div class='step-item'><div class='step-dot'>4</div>&nbsp;Get Report</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Main layout ──────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1.35], gap="large")

# Shared state across cols
preview_image = None
image_bytes = None
temp, humidity = 25, 60

with col1:
    st.markdown("<div class='gl-card-title'>📸 Image Input</div>", unsafe_allow_html=True)

    image_file = st.file_uploader(
        "Upload Leaf Image",
        type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"],
        label_visibility="collapsed",
    )
    st.markdown(
        "<div style='font-size:0.78rem; color:#6ee7b7; margin-top:-0.4rem; margin-bottom:0.8rem;'>"
        "📎 Supports JPG · PNG · WebP · BMP &nbsp;|&nbsp; "
        "Google image? Right-click → Save as JPEG first</div>",
        unsafe_allow_html=True,
    )

    city = st.text_input("🏙️ City for weather data", value="Jaipur", placeholder="e.g. Jaipur, Delhi, Mumbai")

    if image_file:
        try:
            preview_image, image_bytes = prepare_image_bytes(image_file)
            temp, humidity = get_weather(city)

            st.image(preview_image, use_container_width=True)
            st.markdown(
                f"""
                <div style='margin-top:0.8rem;'>
                    <span class='weather-badge'>🌡️ {temp} °C</span>&nbsp;
                    <span class='weather-badge'>💧 {humidity}% humidity</span>&nbsp;
                    <span class='weather-badge'>📍 {city}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except ValueError as e:
            st.error(f"⚠️ {e}")
            image_file = None


with col2:
    st.markdown("<div class='gl-card-title'>🔬 Diagnosis Report</div>", unsafe_allow_html=True)

    if not image_file or preview_image is None:
        st.markdown(
            """
            <div class='gl-card' style='text-align:center; padding: 3rem 1.5rem;'>
                <div style='font-size:3rem;'>🌱</div>
                <div style='color:#6ee7b7 !important; margin-top:0.8rem; font-size:1rem;'>
                    Upload a leaf image on the left to begin
                </div>
                <div style='color:#4b7c5e !important; margin-top:0.5rem; font-size:0.85rem;'>
                    Identifies plant species · Detects disease · Weather-aware advice
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Quick visual check before analysis
        symptom_check = detect_visible_leaf_stress(preview_image)
        stress_label = "⚠️ Visible stress signs detected" if symptom_check["visible_stress"] else "✅ No obvious visible stress"
        stress_color = "#fde68a" if symptom_check["visible_stress"] else "#86efac"
        st.markdown(
            f"""
            <div class='gl-card'>
                <div class='gl-card-title'>🔎 Quick Visual Check</div>
                <div style='color:{stress_color} !important; font-weight:600;'>{stress_label}</div>
                <div style='color:#a7f3d0 !important; font-size:0.85rem; margin-top:0.4rem;'>
                    {symptom_check['summary']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        analyze_clicked = st.button("🔍 Analyze Leaf", use_container_width=True)

        if analyze_clicked:
            with st.spinner("🧬 Running AI analysis…"):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    plant_future = executor.submit(safe_identify_plant, image_bytes)
                    disease_future = executor.submit(safe_identify_disease, image_bytes)
                    plant_result, plant_error = plant_future.result()
                    disease_result, disease_error = disease_future.result()

            # Plant card
            if plant_result:
                conf_pct = int(plant_result["confidence"] * 100)
                st.markdown(
                    f"""
                    <div class='gl-card'>
                        <div class='gl-card-title'>🌿 Plant Identified</div>
                        <div style='font-size:1.15rem; font-weight:700; color:#86efac !important;'>
                            {plant_result['common_name']}
                        </div>
                        <div style='font-size:0.85rem; font-style:italic; color:#a7f3d0 !important;'>
                            {plant_result['scientific_name']}
                        </div>
                        <div style='font-size:0.8rem; color:#6ee7b7 !important; margin-top:0.5rem;'>
                            Confidence: {conf_pct}%
                        </div>
                        <div class='conf-bar-wrap'>
                            <div class='conf-bar-fill' style='width:{conf_pct}%;'></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.warning("⚠️ Plant could not be identified confidently. Try a sharper, well-lit image.")
            if plant_error:
                st.info(f"ℹ️ {plant_error}")

            # Disease card
            if disease_result and disease_result["confidence"] >= 0.30:
                d_conf = int(disease_result["confidence"] * 100)
                st.markdown(
                    f"""
                    <div class='gl-card'>
                        <div class='gl-card-title'>🦠 Disease Detected</div>
                        <div style='font-size:1.1rem; font-weight:700; color:#fde68a !important;'>
                            {disease_result['name']}
                        </div>
                        <div style='font-size:0.8rem; color:#fef08a !important; margin-top:0.4rem;'>
                            Confidence: {d_conf}%
                        </div>
                        <div class='conf-bar-wrap'>
                            <div class='conf-bar-fill'
                                style='width:{d_conf}%; background:linear-gradient(90deg,#f59e0b,#fde68a);'></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif symptom_check["visible_stress"]:
                st.markdown(
                    """
                    <div class='gl-card' style='border-color:rgba(234,179,8,0.3);'>
                        <div class='gl-card-title' style='color:#fbbf24 !important;'>⚠️ Visible Stress</div>
                        <div style='color:#fde68a !important; font-size:0.92rem;'>
                            The leaf shows signs of disease, but exact condition couldn't be named.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class='gl-card'>
                        <div class='gl-card-title'>✅ No Disease Found</div>
                        <div style='color:#a7f3d0 !important; font-size:0.92rem;'>
                            No significant disease detected. Try a closer photo for a better result.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            if disease_error:
                st.info(f"ℹ️ {disease_error}")

            # Full AI report
            st.markdown(
                "<div class='gl-card-title' style='margin-top:1rem;'>📋 Full AI Report</div>",
                unsafe_allow_html=True,
            )
            with st.spinner("✍️ Generating detailed report…"):
                report = analyze_leaf_with_context(
                    preview_image, plant_result, disease_result,
                    city, temp, humidity, symptom_check,
                )
            st.markdown(report)