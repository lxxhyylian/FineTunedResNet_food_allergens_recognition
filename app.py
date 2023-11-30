
import streamlit as st
import subprocess
subprocess.call(["pip", "install", "-r", "./requirements.txt"])
import torch
from skimage.io import imread as imread
from sklearn.utils import resample
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
class FineTunedResNet(nn.Module):
    def __init__(self, num_classes=33):
        super(FineTunedResNet, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.conv1x1 = nn.Conv2d(resnet.fc.in_features, 512, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout(0.5)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=1),
            nn.BatchNorm2d(512),
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.conv1x1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        x = self.bottleneck(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


model = FineTunedResNet()
model.load_state_dict(torch.load('./FineTunedResNet_allergens_model_1e-4.pth', map_location='cpu'))
model.cpu()
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

st.title("lxhyylian-Food Allergens Recognition")
with open('./data/allergens.txt', 'r') as file:
    allergens = [line.strip() for line in file]
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img = transform(img)
    img = img.unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        pred_title = ', '.join(['{} ({:2.1f}%)'.format(allergens[j], 100 * torch.sigmoid(output[0, j]).item())
                            for j, v in enumerate(output.squeeze())
                            if torch.sigmoid(v) > 0.5])

    st.image(img.squeeze(0), caption="Uploaded Image", use_column_width=True)
    st.write("Class predictions:")
    st.write(pred_title)
