# app.py - Streamlit Web Application for MNIST Digit Generator

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import os

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    st.error("PyTorch is not installed. Please install PyTorch to use this app.")
    TORCH_AVAILABLE = False

# Set Streamlit page config
st.set_page_config(
    page_title="MNIST Digit Generator",
    page_icon="🔢",
    layout="wide"
)

# ----------------- Generator Class ----------------- #
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.label_emb = nn.Embedding(num_classes, 50)

        self.model = nn.Sequential(
            nn.Linear(latent_dim + 50, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),

            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_emb = self.label_emb(labels)
        gen_input = torch.cat([noise, label_emb], dim=1)
        img = self.model(gen_input)
        return img.view(img.size(0), 1, 28, 28)

# ----------------- Load Trained Model ----------------- #
@st.cache_resource
def load_model():
    if not TORCH_AVAILABLE:
        return None, None, None

    device = torch.device("cpu")
    latent_dim = 100
    num_classes = 10
    generator = Generator(latent_dim, num_classes)

    model_path = "models/generator.pth"
    if not os.path.exists(model_path):
        st.error("❌ Model file not found: models/generator.pth")
        return None, None, None

    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'generator_state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['generator_state_dict'])
        else:
            generator.load_state_dict(checkpoint)
        generator.eval()

        for module in generator.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval()

        return generator, latent_dim, device
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None, None, None

# ----------------- Generate Digit Images ----------------- #
def generate_digit_images(generator, digit, latent_dim, device, num_samples=5, seed=None):
    if not TORCH_AVAILABLE:
        return None

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    with torch.no_grad():
        noise = torch.randn(num_samples, latent_dim, device=device)
        labels = torch.full((num_samples,), digit, dtype=torch.long, device=device)
        images = generator(noise, labels)
        images = images.cpu().numpy()
        images = (images + 1) / 2  # Normalize from [-1,1] to [0,1]
        return np.clip(images, 0, 1)

# ----------------- Create Image Grid ----------------- #
def create_image_grid(images):
    fig, axes = plt.subplots(1, len(images), figsize=(15, 3))
    for i, img in enumerate(images):
        axes[i].imshow(img[0], cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f'Sample {i+1}', fontsize=10)
        axes[i].axis('off')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return Image.open(buf)

# ----------------- Main App ----------------- #
def main():
    if not TORCH_AVAILABLE:
        st.error("❌ PyTorch not available. Please check your installation.")
        return

    st.title("🔢 MNIST Handwritten Digit Generator")
    st.markdown("Generate handwritten digits using a trained Conditional GAN (cGAN).")

    # Load model
    generator, latent_dim, device = load_model()
    if generator is None:
        return

    # Sidebar controls
    st.sidebar.header("Controls")
    selected_digit = st.sidebar.selectbox("Select digit (0-9):", list(range(10)), index=0)

    use_seed = st.sidebar.checkbox("Use fixed seed")
    seed = None
    if use_seed:
        seed = st.sidebar.number_input("Seed value:", value=42, min_value=0, max_value=9999)

    if 'generate_new' not in st.session_state:
        st.session_state.generate_new = True

    if st.sidebar.button("🎲 Generate Images"):
        st.session_state.generate_new = not st.session_state.generate_new

    # Generate images
    if st.session_state.generate_new:
        with st.spinner("Generating..."):
            images = generate_digit_images(generator, selected_digit, latent_dim, device, seed=seed)
            if images is not None:
                grid = create_image_grid(images)
                st.image(grid, caption=f"5 Samples of Digit {selected_digit}", use_column_width=True)

                st.markdown("### Individual Samples")
                cols = st.columns(5)
                for i, img in enumerate(images):
                    with cols[i]:
                        pil_img = Image.fromarray((img[0] * 255).astype(np.uint8), mode='L')
                        st.image(pil_img, caption=f"Sample {i+1}", use_column_width=True)

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app uses a Conditional GAN trained on the MNIST dataset to generate images of handwritten digits.
    
    - Model: Generator conditioned on digit label
    - Input: 100-dim noise + label
    - Output: 28x28 grayscale images
    """)

if __name__ == "__main__":
    main()
