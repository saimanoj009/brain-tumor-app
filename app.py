import streamlit as st
import numpy as np
import cv2
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile
import os
import base64

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet

# -------------------------------
# Load model & classes
# -------------------------------
model = load_model("brain_tumor_model.h5", compile=False)
model.predict(np.zeros((1, 224, 224, 3)))  # force build

with open("class_names.json", "r") as f:
    class_names = json.load(f)

# -------------------------------
# 🖼️ BACKGROUND IMAGE
# -------------------------------
def get_base64_image(img_path):
    if not os.path.exists(img_path):
        return ""
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.join(BASE_DIR, "assets", "bg.jpg")
img_base64 = get_base64_image(IMG_PATH)

# -------------------------------
# Prediction functions
# -------------------------------
def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return np.reshape(img, (1, 224, 224, 3))

def predict_image(img_path):
    pred = model.predict(preprocess(img_path))[0]
    return class_names[np.argmax(pred)], np.max(pred)

def predict_full(img_path):
    return model.predict(preprocess(img_path))[0]

# -------------------------------
# Grad-CAM
# -------------------------------
def generate_gradcam(img_path, model):
    base_model = model.layers[0]

    last_conv_layer = None
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))

    if last_conv_layer is None:
        return img

    img_norm = img / 255.0
    input_tensor = tf.convert_to_tensor(np.expand_dims(img_norm, axis=0), dtype=tf.float32)

    conv_model = tf.keras.Model(inputs=base_model.input, outputs=last_conv_layer.output)

    with tf.GradientTape() as tape:
        conv_outputs = conv_model(input_tensor)
        preds = model(input_tensor)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        return img

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap.numpy(), 0)
    heatmap /= (np.max(heatmap) + 1e-8)

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return cv2.addWeighted(img, 0.5, heatmap, 0.8, 0)

# -------------------------------
# Bounding Box
# -------------------------------
def draw_bounding_box(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return img

# -------------------------------
# 📄 PDF GENERATION
# -------------------------------
def generate_pdf(label, conf, probs, original_path, heatmap_img):
    pdf_path = "brain_tumor_report.pdf"

    doc = SimpleDocTemplate(pdf_path)
    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("Brain Tumor Detection Report", styles["Title"]))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(f"<b>Prediction:</b> {label.upper()}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Confidence:</b> {conf*100:.2f}%", styles["Normal"]))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("<b>Prediction Breakdown:</b>", styles["Heading3"]))
    for i, p in enumerate(probs):
        elements.append(Paragraph(f"{class_names[i]}: {p:.2f}", styles["Normal"]))

    elements.append(Spacer(1, 20))

    # Save heatmap temp
    heatmap_path = "heatmap.jpg"
    cv2.imwrite(heatmap_path, heatmap_img)

    elements.append(Paragraph("<b>Original MRI Image:</b>", styles["Heading3"]))
    elements.append(RLImage(original_path, width=250, height=250))

    elements.append(Spacer(1, 15))

    elements.append(Paragraph("<b>Tumor Heatmap:</b>", styles["Heading3"]))
    elements.append(RLImage(heatmap_path, width=250, height=250))

    doc.build(elements)

    return pdf_path

# -------------------------------
# UI CONFIG
# -------------------------------
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

# -------------------------------
# BACKGROUND CSS
# -------------------------------
if img_base64:
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .stApp::before {{
        content: "";
        position: fixed;
        top:0; left:0;
        width:100%; height:100%;
        background: rgba(0,0,0,0.75);
        z-index: -1;
    }}
    .block-container {{
        background: rgba(0, 0, 0, 0.7);
        padding: 2rem;
        border-radius: 20px;
        backdrop-filter: blur(12px);
    }}
    h1 {{
        text-align:center;
        font-size:50px;
        background: linear-gradient(90deg,#00f2fe,#4facfe);
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent;
    }}
    </style>
    """, unsafe_allow_html=True)

# -------------------------------
# UI
# -------------------------------

st.markdown("""
<style>

/* 🌌 Global Text */
.stApp {
    color: white !important;
}

/* ✨ Glass Card */
.card {
    background: rgba(0, 0, 0, 0.65);
    padding: 15px;
    border-radius: 15px;
    margin-bottom: 12px;
    border: 1px solid rgba(0,255,255,0.2);
    box-shadow: 0 0 15px rgba(0,255,255,0.15);
    text-align: center;
}

/* 🧠 Section Titles */
.card h3 {
    color: #00f2fe;
    font-size: 20px;
    margin-bottom: 10px;
}

/* 🔥 Confidence Highlight */
.confidence {
    font-size: 38px;
    font-weight: bold;
    color: #00ffcc;
    text-shadow: 0 0 12px rgba(0,255,255,0.9);
}

/* 🖼️ Image Effects */
img {
    border-radius: 12px;
    transition: 0.3s;
}

img:hover {
    transform: scale(1.05);
    box-shadow: 0 0 25px rgba(0,255,255,0.6);
}

/* 📊 Breakdown Box */
.breakdown {
    padding: 10px;
    background: rgba(255,255,255,0.05);
    margin: 6px;
    border-radius: 10px;
    font-size: 16px;
    color: white;
    border-left: 3px solid #00f2fe;
}

/* 🔘 Upload Box */
[data-testid="stFileUploader"] {
    border: 2px dashed #00f2fe;
    border-radius: 12px;
    padding: 15px;
    background: rgba(0, 0, 0, 0.6);
}

/* Fix upload inner area */
[data-testid="stFileUploader"] section {
    background: transparent !important;
    color: white !important;
}

/* Upload button */
[data-testid="stFileUploader"] button {
    background-color: #00f2fe !important;
    color: black !important;
    font-weight: bold;
    border-radius: 8px;
    border: none;
}

[data-testid="stFileUploader"] button:hover {
    background-color: #4facfe !important;
    color: white !important;
}

/* Hide default helper text */
[data-testid="stFileUploader"] small {
    display: none !important;
}

/* Upload label */
[data-testid="stFileUploader"] label {
    color: white !important;
    font-weight: bold;
    font-size: 16px;
}

/* 📄 Download Button */
.stDownloadButton>button {
    background: linear-gradient(45deg, #00f2fe, #4facfe);
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
    font-weight: bold;
    transition: 0.3s;
    box-shadow: 0 0 10px #00f2fe;
}

.stDownloadButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 25px #00f2fe;
}

/* ⚠️ Alerts */
.stAlert {
    border-radius: 10px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# -------------------------------
# 🧠 HEADER
# -------------------------------
st.markdown("<h1 style='text-align:center;'>🧠 Brain Tumor Detection System</h1>", unsafe_allow_html=True)

st.markdown("<p style='text-align:center; color:#ccc;'>Upload MRI scan for detection</p>", unsafe_allow_html=True)


# -------------------------------
# 📤 FILE UPLOAD
# -------------------------------
file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

st.markdown(
    "<p style='text-align:center; color:#00f2fe;'>Supported formats: JPG, PNG, JPEG</p>",
    unsafe_allow_html=True
)


# -------------------------------
# ⚙️ PROCESSING
# -------------------------------
if file:
    img = Image.open(file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        img.save(temp.name)

        label, conf = predict_image(temp.name)
        probs = predict_full(temp.name)

        heatmap_img = generate_gradcam(temp.name, model)
        heatmap_img = draw_bounding_box(heatmap_img)

    st.success("✅ Analysis Complete")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='card'><h3>🖼️ Original Image</h3></div>", unsafe_allow_html=True)
        st.image(img, width=300)

    with col2:
        st.markdown("<div class='card'><h3>🔥 Tumor Heatmap</h3></div>", unsafe_allow_html=True)
        st.image(heatmap_img, width=300)

    # Prediction
    if label.lower() == "notumor":
        st.success(f"✅ Prediction: {label.upper()}")
    else:
        st.error(f"🚨 Prediction: {label.upper()}")

    # Confidence
    st.markdown(f"""
    <div class='card'>
        <h3>📊 Confidence Score</h3>
        <div class='confidence'>{conf*100:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # Breakdown
    st.markdown("<div class='card'><h3>📌 Prediction Breakdown</h3></div>", unsafe_allow_html=True)

    for i, p in enumerate(probs):
        st.markdown(f"""
        <div class='breakdown'>
            <b>{class_names[i].upper()}</b> : {p:.2f}
        </div>
        """, unsafe_allow_html=True)

    st.warning("⚠️ AI-based result. Not a medical diagnosis.")

    # PDF
    pdf_file = generate_pdf(label, conf, probs, temp.name, heatmap_img)

    with open(pdf_file, "rb") as f:
        st.download_button(
            label="📄 Download Full Report",
            data=f,
            file_name="brain_tumor_report.pdf",
            mime="application/pdf"
        )


# -------------------------------
# 📌 CLEAN FOOTER
# -------------------------------
st.markdown("""
<hr style="border:1px solid rgba(255,255,255,0.1); margin-top:40px;">

<p style='text-align:center; font-size:13px; color:#aaa;'>
© 2026 Brain Tumor Detection System • Sai Manoj • 
<a href="mailto:saimanoj0914@gmail.com" style="color:#00f2fe; text-decoration:none;">
saimanoj0914@gmail.com</a>
</p>
""", unsafe_allow_html=True)
