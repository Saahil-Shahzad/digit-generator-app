# app.py

import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Generator class (same as in training)
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.view(-1, 1, 28, 28)

# Load generator
device = torch.device("cpu")
generator = Generator(100, 10).to(device)
generator.load_state_dict(torch.load("models/generator.pth", map_location=device))
generator.eval()

# Streamlit UI
st.title("MNIST Handwritten Digit Generator")
digit = st.selectbox("Choose a digit to generate (0-9):", list(range(10)))

if st.button("Generate 5 images"):
    z = torch.randn(5, 100)
    labels = torch.full((5,), digit, dtype=torch.long)
    with torch.no_grad():
        generated = generator(z, labels).detach().cpu()

    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axs[i].imshow(generated[i][0], cmap="gray")
        axs[i].axis('off')
    st.pyplot(fig)
