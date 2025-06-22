# app.py - Streamlit Web Application

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

# Set page config
st.set_page_config(
    page_title="MNIST Digit Generator",
    page_icon="🔢",
    layout="wide"
)

# Generator class (same as in training script)
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Label embedding
        self.label_emb = nn.Embedding(num_classes, 50)
        
        # Main generator network
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
        # Get label embeddings
        label_emb = self.label_emb(labels)
        
        # Concatenate noise and label embeddings
        gen_input = torch.cat([noise, label_emb], dim=1)
        
        # Generate image
        img = self.model(gen_input)
        img = img.view(img.size(0), 1, 28, 28)
        
        return img

@st.cache_resource
def load_model():
    """Load the trained generator model with PyTorch 2.6+ compatibility"""
    if not TORCH_AVAILABLE:
        st.error("PyTorch is not available.")
        return None, None, None
        
    device = torch.device("cpu")
    
    # Model parameters
    latent_dim = 100
    num_classes = 10
    
    # Initialize generator
    generator = Generator(latent_dim, num_classes)
    
    # Try to load the model
    model_path = "models/generator.pth"
    if os.path.exists(model_path):
        try:
            # Method 1: Try with weights_only=False (trusted source)
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
            if isinstance(checkpoint, dict) and 'generator_state_dict' in checkpoint:
                generator.load_state_dict(checkpoint['generator_state_dict'])
            else:
                generator.load_state_dict(checkpoint)
            
            generator.eval()
            return generator, latent_dim, device
            
        except Exception as e1:
            try:
                # Method 2: Try with safe globals (for newer PyTorch)
                import torch.serialization
                with torch.serialization.safe_globals([Generator]):
                    checkpoint = torch.load(model_path, map_location=device)
                    if isinstance(checkpoint, dict) and 'generator_state_dict' in checkpoint:
                        generator.load_state_dict(checkpoint['generator_state_dict'])
                    else:
                        generator.load_state_dict(checkpoint)
                    generator.eval()
                    return generator, latent_dim, device
            except Exception as e2:
                st.error(f"Error loading model: {e1}")
                st.error(f"Fallback method also failed: {e2}")
                st.error("Please retrain the model with the updated training script.")
                return None, None, None
    else:
        st.error("Model file not found. Please train the model first.")
        return None, None, None

def generate_digit_images(generator, digit, latent_dim, device, num_samples=5):
    """Generate multiple images of a specific digit"""
    if not TORCH_AVAILABLE:
        return None
        
    with torch.no_grad():
        # Create different noise vectors for diversity
        noise = torch.randn(num_samples, latent_dim, device=device)
        labels = torch.full((num_samples,), digit, device=device, dtype=torch.long)
        
        # Generate images
        generated_imgs = generator(noise, labels)
        
        # Convert to numpy and denormalize
        generated_imgs = generated_imgs.cpu().numpy()
        generated_imgs = (generated_imgs + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
        generated_imgs = np.clip(generated_imgs, 0, 1)  # Ensure values are in [0, 1]
        
        return generated_imgs

def create_image_grid(images):
    """Create a grid of images for display"""
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    
    for i, img in enumerate(images):
        axes[i].imshow(img[0], cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f'Sample {i+1}', fontsize=12)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)

# Main app
def main():
    if not TORCH_AVAILABLE:
        st.error("❌ PyTorch is not available. Please check your requirements.txt and redeploy.")
        return
        
    st.title("🔢 MNIST Handwritten Digit Generator")
    st.markdown("Generate handwritten digits using a trained Conditional GAN!")
    
    # Load model
    generator, latent_dim, device = load_model()
    
    if generator is None:
        st.error("❌ Model not available. Please train the model first using the training script.")
        st.markdown("### Instructions:")
        st.markdown("1. Run the training script to train the GAN model")
        st.markdown("2. Make sure the model is saved in the `models/` directory")
        st.markdown("3. Refresh this page")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar for controls
    st.sidebar.header("Generation Controls")
    
    # Digit selection
    selected_digit = st.sidebar.selectbox(
        "Select digit to generate (0-9):",
        options=list(range(10)),
        index=0
    )
    
    # Generation button
    if st.sidebar.button("🎲 Generate New Images", type="primary"):
        st.session_state.generate_new = True
    
    # Add seed control for reproducibility
    use_seed = st.sidebar.checkbox("Use fixed seed for reproducible results")
    if use_seed:
        seed = st.sidebar.number_input("Seed value", value=42, min_value=0, max_value=9999)
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Main content area
    st.markdown(f"### Generated Images of Digit: **{selected_digit}**")
    
    # Generate images
    if 'generate_new' not in st.session_state:
        st.session_state.generate_new = True
    
    if st.session_state.generate_new:
        with st.spinner("Generating images..."):
            try:
                # Generate 5 images of the selected digit
                generated_images = generate_digit_images(
                    generator, selected_digit, latent_dim, device, num_samples=5
                )
                
                if generated_images is not None:
                    # Create and display image grid
                    image_grid = create_image_grid(generated_images)
                    st.image(image_grid, caption=f"5 generated samples of digit {selected_digit}")
                    
                    # Display individual images in columns
                    st.markdown("### Individual Samples:")
                    cols = st.columns(5)
                    for i, img in enumerate(generated_images):
                        with cols[i]:
                            # Convert numpy array to PIL Image
                            pil_img = Image.fromarray((img[0] * 255).astype(np.uint8), mode='L')
                            st.image(pil_img, caption=f"Sample {i+1}", use_column_width=True)
                    
                    st.session_state.generate_new = False
                else:
                    st.error("Failed to generate images.")
                
            except Exception as e:
                st.error(f"Error generating images: {e}")
    
    # Information section
    st.markdown("---")
    st.markdown("### About this App")
    st.markdown("""
    This application uses a **Conditional Generative Adversarial Network (cGAN)** trained on the MNIST dataset 
    to generate handwritten digits. The model was trained from scratch using PyTorch.
    
    **Key Features:**
    - Generate 5 different samples of any digit (0-9)
    - Each generation uses different random noise for variety
    - Model trained on 28x28 grayscale images
    - Uses label conditioning to control which digit is generated
    """)
    
    # Model information
    with st.expander("🧠 Model Architecture Details"):
        st.markdown("""
        **Generator:**
        - Input: 100-dim noise vector + digit label embedding
        - Architecture: Linear layers with BatchNorm and ReLU
        - Output: 28x28 grayscale image
        
        **Training:**
        - Dataset: MNIST (60,000 training images)
        - Loss: Binary Cross Entropy
        - Optimizer: Adam with different learning rates for G and D
        - Training time: ~50 epochs on single T4 GPU
        """)

if __name__ == "__main__":
    main()
