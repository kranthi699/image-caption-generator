"""
Plant Disease Detection - Streamlit app
Run:  streamlit run app.py
Requires outputs/plant_disease_model.keras and outputs/class_names.json (created by train.py).
"""
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = "outputs/plant_disease_model.keras"
META_PATH = "outputs/class_names.json"

st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="centered")

@st.cache_resource
def load():
    model = tf.keras.models.load_model(MODEL_PATH)
    meta = json.load(open(META_PATH))
    return model, meta["class_names"], meta["img_size"]

def pretty(name: str) -> str:
    return name.replace("___", " - ").replace("__", " - ").replace("_", " ")

st.title("🌿 Plant Disease Detection")
st.write("Upload a clear photo of a single leaf and the CNN will predict its health status.")

try:
    model, class_names, img_size = load()
except Exception as e:
    st.error("Model not found. Train it first with `python train.py --data_dir <path>`.")
    st.stop()

file = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"])
if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded leaf", use_container_width=True)

    arr = np.expand_dims(np.array(img.resize((img_size, img_size)), dtype="float32"), 0)
    probs = model.predict(arr, verbose=0)[0]
    top = np.argsort(probs)[::-1][:3]

    label, conf = pretty(class_names[top[0]]), probs[top[0]] * 100
    if "healthy" in label.lower():
        st.success(f"**{label}**  —  confidence {conf:.1f}%")
    else:
        st.warning(f"**{label}**  —  confidence {conf:.1f}%")
    if conf < 60:
        st.info("Low confidence: try a clearer, well-lit, close-up photo of one leaf.")

    st.subheader("Top 3 predictions")
    for i in top:
        st.progress(float(probs[i]), text=f"{pretty(class_names[i])}: {probs[i]*100:.1f}%")
