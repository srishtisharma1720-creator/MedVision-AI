import torch
import gdown
import os
import streamlit as st
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="MedVision AI",
    page_icon="🩺",
    layout="wide"
)

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# MODEL DOWNLOAD
# =========================
MODEL_PATH = "pneumonia_model.pth"
file_id = "1V_FLHVkRt526jcSI42bdUG3ncsjmgVIj"

url = f"https://drive.google.com/uc?export=download&id={file_id}"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(url, MODEL_PATH, quiet=False)

# =========================
# MODEL ARCHITECTURE
# =========================
class PneumoniaCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# =========================
# LOAD MODEL
# =========================
model = PneumoniaCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

classes = ["NORMAL", "PNEUMONIA"]

# =========================
# UI DESIGN
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0f172a,#1e293b);
}

.hero {
    text-align:center;
    padding:20px;
}

.hero h1 {
    font-size:65px;
    font-weight:900;
    background: linear-gradient(90deg,#38bdf8,#818cf8,#c084fc);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.hero h3 {
    color:#cbd5e1;
}

.pred-normal {
    background: linear-gradient(90deg,#22c55e,#16a34a);
    padding:20px;
    border-radius:15px;
    color:white;
    font-size:32px;
    font-weight:bold;
    text-align:center;
}

.pred-pneumonia {
    background: linear-gradient(90deg,#ef4444,#dc2626);
    padding:20px;
    border-radius:15px;
    color:white;
    font-size:32px;
    font-weight:bold;
    text-align:center;
}

.card {
    background: rgba(255,255,255,0.08);
    padding:15px;
    border-radius:15px;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HERO
# =========================
st.markdown("""
<div class="hero">
<h1>🩺 MedVision AI</h1>
<h3>Chest X-Ray Pneumonia Detection System</h3>
</div>
""", unsafe_allow_html=True)

# =========================
# UPLOAD IMAGE
# =========================
uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)[0]
        confidence, pred = torch.max(probs, 0)

    prediction = classes[pred.item()]
    confidence_pct = confidence.item() * 100

    st.markdown("---")

    # RESULT
    if prediction == "NORMAL":
        st.markdown('<div class="pred-normal">✅ NORMAL</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="pred-pneumonia">⚠️ PNEUMONIA</div>', unsafe_allow_html=True)

    # CONFIDENCE GAUGE
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence_pct,
        title={"text": "Confidence"},
        gauge={"axis": {"range": [0, 100]}}
    ))

    st.plotly_chart(fig, use_container_width=True)

    # PROBABILITY CHART
    df = pd.DataFrame({
        "Class": classes,
        "Probability": [probs[0].item()*100, probs[1].item()*100]
    })

    fig2 = px.bar(df, x="Class", y="Probability", text="Probability")
    st.plotly_chart(fig2, use_container_width=True)

    # REPORT
    st.markdown(f"""
    <div class="card">
    <h2>AI Report</h2>
    <b>Diagnosis:</b> {prediction}<br>
    <b>Confidence:</b> {confidence_pct:.2f}%<br>
    <b>Note:</b> For educational use only
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Built with PyTorch + Streamlit + CNN")
