import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same CNN architecture as train.py
class PneumoniaCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
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

# Load model
model = PneumoniaCNN().to(device)
model.load_state_dict(
    torch.load("pneumonia_model.pth", map_location=device)
)

model.eval()

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Image path
image_path = "sample.jpg"   # Change to your image

# Load image
image = Image.open(image_path).convert("RGB")

# Preprocess
img = transform(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    outputs = model(img)
    _, predicted = torch.max(outputs, 1)

classes = ["NORMAL", "PNEUMONIA"]

print("Prediction:", classes[predicted.item()])