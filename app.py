import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Digit Recogniser",
    page_icon="✏️",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=DM+Mono:wght@300;400;500&display=swap');

:root {
    --bg:       #0e0e10;
    --surface:  #18181c;
    --border:   #2a2a30;
    --accent:   #e8ff47;
    --correct:  #47ff8a;
    --text:     #f0f0f0;
    --muted:    #888;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}

/* Hide only the footer */
footer { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #131316 !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
    color: var(--text) !important;
}
[data-testid="stSidebarContent"] {
    padding-top: 1.5rem;
}

/* Header */
.page-header {
    padding: 1.8rem 0 0.5rem;
}
.page-header h1 {
    font-family: 'Inter', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 0.2rem;
    letter-spacing: -0.01em;
}
.page-header p {
    color: var(--muted);
    font-size: 0.9rem;
    margin: 0;
}

/* Cover image wrapper */
.cover-wrap {
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 1.6rem;
    border: 1px solid var(--border);
}
.cover-wrap img {
    width: 100%;
    display: block;
}

/* Prediction badge */
.pred-badge {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1.5rem;
    padding: 1.8rem;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-top: 0.8rem;
}
.pred-digit {
    font-family: 'Inter', sans-serif;
    font-size: 5.5rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
}
.pred-meta {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
}
.pred-label {
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.pred-conf {
    font-family: 'Inter', sans-serif;
    font-size: 1.9rem;
    font-weight: 600;
    color: var(--correct);
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.2rem 0;
}

/* Upload zone */
[data-testid="stFileUploaderDropzone"] {
    background: var(--surface) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 10px !important;
    color: var(--muted) !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent) !important;
}

/* Sidebar info boxes */
.info-block {
    background: #1e1e24;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.8rem;
    font-size: 0.82rem;
    line-height: 1.6;
}
.info-block .info-title {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--accent);
    margin-bottom: 0.4rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load model (cached so it only loads once) ─────────────────────────────────
@st.cache_resource
def load_model():
    # Expects mnist_model.keras to sit next to app.py
    model = tf.keras.models.load_model("mnist_model.keras")
    return model

model = load_model()

# ── Helper: preprocess an uploaded PIL image ──────────────────────────────────
def preprocess(img: Image.Image) -> np.ndarray:
    # Convert to greyscale
    img = ImageOps.grayscale(img)

    # Invert if background is light (MNIST is white digit on black)
    arr = np.array(img)
    if arr.mean() > 127:
        img = ImageOps.invert(img)

    # Resize to 28x28 to match model input
    img = img.resize((28, 28), Image.LANCZOS)

    # Normalise to [0,1] and flatten to (1, 784)
    arr = np.array(img, dtype="float32") / 255.0
    return arr.reshape(1, 784)

# ── Helper: confidence bar chart ──────────────────────────────────────────────
def confidence_chart(probs: np.ndarray, predicted: int):
    digits = list(range(10))
    colors = ["#e8ff47" if i == predicted else "#2a2a30" for i in digits]
    edge_colors = ["#e8ff47" if i == predicted else "#444" for i in digits]

    fig, ax = plt.subplots(figsize=(9, 3))
    fig.patch.set_facecolor("#18181c")
    ax.set_facecolor("#18181c")

    bars = ax.bar(
        digits,
        probs * 100,
        color=colors,
        edgecolor=edge_colors,
        linewidth=0.8,
        width=0.65,
        zorder=3,
    )

    # Label bars with their percentage value
    for bar, val in zip(bars, probs * 100):
        if val > 1.5:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#f0f0f0",
                fontfamily="monospace",
            )

    ax.set_xticks(digits)
    ax.set_xticklabels([str(d) for d in digits], color="#888", fontsize=10)
    ax.set_ylabel("Confidence (%)", color="#888", fontsize=9)
    ax.set_ylim(0, 115)
    ax.tick_params(axis="y", colors="#888", labelsize=8)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.yaxis.grid(True, color="#2a2a30", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    highlight = mpatches.Patch(color="#e8ff47", label=f"Predicted: {predicted}")
    ax.legend(handles=[highlight], loc="upper right", framealpha=0, labelcolor="#f0f0f0", fontsize=9)

    plt.tight_layout()
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✏️ About")
    st.markdown("""
    <div class="info-block">
        <div class="info-title">Model</div>
        Dense Neural Network trained on MNIST. Three hidden layers (512 → 256 → 128) with dropout, softmax output over 10 digit classes.
    </div>
    <div class="info-block">
        <div class="info-title">Dataset</div>
        MNIST — 60,000 training images and 10,000 test images of handwritten digits (0–9), each 28×28 pixels.
    </div>
    <div class="info-block">
        <div class="info-title">How to use</div>
        Upload any image of a handwritten digit. The app will preprocess it to match the model's expected input and return the predicted class with confidence scores.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color:#555; line-height:1.6;">
        Built with TensorFlow · Keras · Streamlit
    </div>
    """, unsafe_allow_html=True)

# ── Main content ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>Handwritten Digit Recogniser</h1>
    <p>Upload an image and the model will predict the digit with confidence scores</p>
</div>
""", unsafe_allow_html=True)

# Cover image below the header
st.markdown('<div class="cover-wrap">', unsafe_allow_html=True)
st.image("cover.jpg", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# File uploader
uploaded = st.file_uploader(
    "Upload a handwritten digit image",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
    label_visibility="collapsed",
)

if uploaded is not None:
    img = Image.open(uploaded)

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.markdown("**Uploaded image**")
        st.image(img, use_container_width=True)

    with col2:
        # Run inference
        processed = preprocess(img)
        probs = model.predict(processed, verbose=0)[0]
        predicted = int(np.argmax(probs))
        confidence = float(probs[predicted]) * 100

        st.markdown("**Prediction**")
        st.markdown(f"""
        <div class="pred-badge">
            <div class="pred-digit">{predicted}</div>
            <div class="pred-meta">
                <span class="pred-label">Predicted digit</span>
                <span class="pred-conf">{confidence:.1f}%</span>
                <span class="pred-label">confidence</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("**Confidence across all digit classes**")

    fig = confidence_chart(probs, predicted)
    st.pyplot(fig)
    plt.close(fig)

else:
    st.info("Upload a PNG or JPG image of a handwritten digit (0–9) to get started.", icon="☝️")