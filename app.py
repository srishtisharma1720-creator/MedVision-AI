import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="MedVision AI",
    page_icon="🩺",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#0f172a,#1e293b);
}

.hero{
    text-align:center;
    padding:20px;
}

.hero h1{
    font-size:70px;
    font-weight:900;
    margin-bottom:0;
    background: linear-gradient(90deg,#38bdf8,#818cf8,#c084fc);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.hero h3{
    color:#cbd5e1;
    font-weight:400;
}

.card{
    background:rgba(255,255,255,0.08);
    padding:20px;
    border-radius:20px;
    text-align:center;
    backdrop-filter: blur(10px);
}

.metric-value{
    font-size:32px;
    font-weight:bold;
    color:#38bdf8;
}

.metric-title{
    color:white;
}

.pred-normal{
    background:linear-gradient(90deg,#22c55e,#16a34a);
    padding:25px;
    border-radius:20px;
    text-align:center;
    color:white;
    font-size:40px;
    font-weight:bold;
}

.pred-pneumonia{
    background:linear-gradient(90deg,#ef4444,#dc2626);
    padding:25px;
    border-radius:20px;
    text-align:center;
    color:white;
    font-size:40px;
    font-weight:bold;
}

.report{
    background:rgba(255,255,255,0.08);
    padding:25px;
    border-radius:20px;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# MODEL
# ==========================================
class PneumoniaCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*28*28,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,2)
        )

    def forward(self,x):
        x=self.conv_layers(x)
        x=self.fc_layers(x)
        return x

# ==========================================
# LOAD MODEL
# ==========================================
device = torch.device("cpu")

model = PneumoniaCNN()

model.load_state_dict(
    torch.load(
        "pneumonia_model.pth",
        map_location=device
    )
)

model.eval()

# ==========================================
# TRANSFORM
# ==========================================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

classes = ["NORMAL","PNEUMONIA"]

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("🩺 MedVision AI")

st.sidebar.markdown("""
### About

AI-powered chest X-ray analysis platform.

**Framework:** PyTorch

**Model:** CNN

**Dataset:** Chest X-Ray Pneumonia Dataset

**Classes:**
- NORMAL
- PNEUMONIA
""")

# ==========================================
# HERO
# ==========================================
st.markdown("""
<div class="hero">
<h1>🩺 MedVision AI</h1>
<h3>Next-Generation Chest X-Ray Analysis Platform</h3>
</div>
""", unsafe_allow_html=True)

# ==========================================
# KPI CARDS
# ==========================================
c1,c2,c3,c4 = st.columns(4)

cards = [
    ("Model","CNN"),
    ("Framework","PyTorch"),
    ("Classes","2"),
    ("Domain","Healthcare AI")
]

for col,(title,val) in zip([c1,c2,c3,c4],cards):
    with col:
        st.markdown(f"""
        <div class="card">
        <div class="metric-value">{val}</div>
        <div class="metric-title">{title}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")

# ==========================================
# UPLOAD
# ==========================================
uploaded_file = st.file_uploader(
    "📤 Upload Chest X-Ray Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1,col2 = st.columns([1.3,1])

    with col1:
        st.image(
            image,
            caption="Uploaded X-Ray",
            use_container_width=True
        )

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output,dim=1)[0]
        confidence,pred = torch.max(probs,0)

    prediction = classes[pred.item()]
    confidence_pct = confidence.item()*100

    with col2:

        if prediction == "NORMAL":
            st.markdown(
                '<div class="pred-normal">✅ NORMAL</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="pred-pneumonia">⚠️ PNEUMONIA</div>',
                unsafe_allow_html=True
            )

        st.write("")

        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence_pct,
            title={"text":"Confidence"},
            gauge={
                "axis":{"range":[0,100]}
            }
        ))

        fig.update_layout(
            height=300,
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    # Probability Chart
    st.subheader("📊 Prediction Probabilities")

    prob_df = pd.DataFrame({
        "Class":classes,
        "Probability":[
            probs[0].item()*100,
            probs[1].item()*100
        ]
    })

    fig2 = px.bar(
        prob_df,
        x="Class",
        y="Probability",
        text="Probability"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.markdown("---")

    # Diagnostic Report
    st.markdown(f"""
    <div class="report">

    <h2>🤖 AI Diagnostic Report</h2>

    <hr>

    <b>Diagnosis:</b> {prediction}

    <br><br>

    <b>Confidence:</b> {confidence_pct:.2f}%

    <br><br>

    <b>Model:</b> CNN Deep Learning Network

    <br><br>

    <b>Interpretation:</b><br>

    The uploaded chest X-ray was analyzed by a Convolutional
    Neural Network trained on chest radiography images.

    This prediction is intended for educational and research
    purposes and should not replace professional medical advice.

    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption(
    "🚀 Developed with PyTorch • Streamlit • Deep Learning • Computer Vision"
)