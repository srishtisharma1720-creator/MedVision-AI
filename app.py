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


st.set_page_config(
    page_title="MedVision AI",
    page_icon="🩺",
    layout="wide"
)


with st.sidebar:
    st.title("🩺 MedVision AI")
    st.write("Chest X-Ray Pneumonia Detection")
    st.markdown("---")

    st.info("Upload an X-ray image to get AI prediction.")
    st.markdown("### Model Info")
    st.write("CNN-based Deep Learning Model")
    st.write("Classes: NORMAL | PNEUMONIA")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_PATH = "pneumonia_model.pth"
file_id = "1V_FLHVkRt526jcSI42bdUG3ncsjmgVIj"
url = f"https://drive.google.com/uc?export=download&id={file_id}"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(url, MODEL_PATH, quiet=False)


class PneumoniaCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
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

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


model = PneumoniaCNN()
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)

model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

classes = ["NORMAL", "PNEUMONIA"]


st.markdown("""
<div style="text-align:center;">
<h1 style="font-size:55px;
background: linear-gradient(90deg,#38bdf8,#818cf8,#c084fc);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;">
🩺 MedVision AI
</h1>
<h3 style="color:#cbd5e1;">AI-powered Chest X-Ray Pneumonia Detection</h3>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

col1, col2, col3 = st.columns([1,2,1])

with col2:

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

        
        if prediction == "NORMAL":
            st.success("✅ NORMAL")
        else:
            st.error("⚠️ PNEUMONIA DETECTED")

        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence_pct,
            title={"text": "Confidence"},
            gauge={"axis": {"range": [0, 100]}}
        ))
        st.plotly_chart(fig, use_container_width=True)

        
        df = pd.DataFrame({
            "Class": classes,
            "Probability": [probs[0].item()*100, probs[1].item()*100]
        })

        fig2 = px.bar(df, x="Class", y="Probability", text="Probability")
        st.plotly_chart(fig2, use_container_width=True)

        
        st.markdown(f"""
        <div style="
            background: rgba(255,255,255,0.08);
            padding:20px;
            border-radius:15px;
            text-align:center;">
            <h3>AI Report</h3>
            <b>Diagnosis:</b> {prediction}<br>
            <b>Confidence:</b> {confidence_pct:.2f}%<br>
            <b>Note:</b> For educational use only
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Built with PyTorch + Streamlit + CNN")
