import streamlit as st
import torch
import matplotlib.pyplot as plt
from torchvision.models import (
    alexnet, AlexNet_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    mnasnet0_5, MNASNet0_5_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    regnet_x_1_6gf, RegNet_X_1_6GF_Weights
)
from torchvision.io import decode_image
from PIL import Image
import torch.nn as nn

# Function to load a model from file
def load_model_from_file(uploaded_file):
    try:
        model = torch.load(uploaded_file, map_location=torch.device('cpu'))
        model.eval()
        return model, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Function to load a predefined model
def load_predefined_model(model_name):
    try:
        if model_name == "AlexNet":
            weights = AlexNet_Weights.IMAGENET1K_V1
            model = alexnet(weights=weights)
        elif model_name == "EfficientNet_B0":
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            model = efficientnet_b0(weights=weights)
        elif model_name == "MNASNet0_5":
            weights = MNASNet0_5_Weights.IMAGENET1K_V1
            model = mnasnet0_5(weights=weights)
        elif model_name == "MobileNet_V3_Small":
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            model = mobilenet_v3_small(weights=weights)
        elif model_name == "RegNet_X_1_6GF":
            weights = RegNet_X_1_6GF_Weights.IMAGENET1K_V1
            model = regnet_x_1_6gf(weights=weights)
        else:
            raise ValueError("Unknown model name")

        model.eval()
        return model, weights
    except Exception as e:
        st.error(f"Error loading predefined model: {str(e)}")
        return None, None

def visualize_kernels(model):
    st.write("Model Architecture:")
    st.write(model)

    weights = dict()
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            weights[name] = module.weight

    if not weights:
        st.warning("No convolutional layers found in the model!")
        return

    selected_name = st.selectbox("Select a layer:", list(weights.keys()))
    kernel = weights[selected_name].detach().numpy()

    if len(kernel.shape) == 4:
        m, n, s, _ = kernel.shape
        fig, axes = plt.subplots(m, n, figsize=(n, m))
        for i in range(m):
            for j in range(n):
                if kernel.shape[-1] == 1:
                    axes[i, j].imshow(kernel[i, j, :, 0], cmap='gray')
                else:
                    axes[i, j].imshow(kernel[i, j, :, :])
                axes[i, j].axis('off')
    elif len(kernel.shape) == 3:
        st.warning("Unusual kernel shape detected. Trying to visualize anyway...")
        m, n, s = kernel.shape
        fig, axes = plt.subplots(m, 1, figsize=(3, m))
        for i in range(m):
            axes[i].imshow(kernel[i, :, :], cmap='gray')
            axes[i].axis('off')
    else:
        st.error(f"Unsupported kernel shape: {kernel.shape}")
        return

    st.pyplot(fig)

def main():
    st.sidebar.title("CNN Kernel Visualization")

    option = st.sidebar.radio(
        "Choose model source:",
        ("Use predefined model", "Upload custom model")
    )

    model = None
    weights = None

    if option == "Use predefined model":
        model_name = st.sidebar.selectbox(
            "Select a predefined model:",
            ["AlexNet", "EfficientNet_B0", "MNASNet0_5", "MobileNet_V3_Small", "RegNet_X_1_6GF"]
        )
        model, weights = load_predefined_model(model_name)
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload your PyTorch model (.pt or .pth file)",
            type=["pt", "pth"]
        )
        if uploaded_file is not None:
            model, weights = load_model_from_file(uploaded_file)

    st.title("Visualization Page")
    if model is not None:
        visualize_kernels(model)

if __name__ == "__main__":
    main()
