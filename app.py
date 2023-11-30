
import streamlit as st
import subprocess
subprocess.call(["pip", "install", "-r", "./requirements.txt"])
import torch
if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.ones(1, device=device)
    print (x)
else:
    device = torch.device("cpu")
    print ("MPS device not found.")
from skimage.io import imread as imread
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import imgaug.augmenters as iaa
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import tqdm
import tqdm.notebook as tqdm
from torch.nn.utils import clip_grad_norm_
import time
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import textwrap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, classification_report



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
model_file = './FineTunedResNet_allergens_model_1e-4.pth'
model.load_state_dict(torch.load(model_file))
model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

st.title("lxhyylian-Food Allergens Recognition")
with open('./allergens.txt', 'r') as file:
    allergens = [line.strip() for line in file]
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img = transform(img)
    img = img.unsqueeze(0)

    with torch.no_grad():
        img = img.to(device)
        output = model(img)
        pred_title = ', '.join(['{} ({:2.1f}%)'.format(allergens[j], 100 * torch.sigmoid(output[0, j]).item())
                            for j, v in enumerate(output.squeeze())
                            if torch.sigmoid(v) > 0.5])

    st.image(img.squeeze(0).cpu().permute(1, 2, 0), caption="Uploaded Image", use_column_width=True)
    st.write("Class predictions:")
    st.write(pred_title)