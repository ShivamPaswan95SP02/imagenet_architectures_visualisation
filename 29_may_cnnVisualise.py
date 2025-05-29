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

# Model---------
def load_model(model_name):
    # Load the pre-trained model based on the model_name
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

# list layer---------------
def list_layers(model):
    # List all layers of the model
    layers = []
    for name, layer in model.named_children():
        layers.append(name)
    return layers



import torch.nn as nn
def main(model_name):
    # Load the model
    model, weights = load_model(model_name)
    model.eval()

    st.write(model)

    weights = dict()

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            kernel = module.weight
            weights[name] = module.weight
    
    selected_name = st.selectbox(
        "Select a layer:", weights.keys())

    kernel = weights[selected_name].detach().numpy()

    m, n, s, _ = kernel.shape

    fig, axes = plt.subplots(m, n, figsize=(n, m))
    for i in range(m):
        for j in range(n):
            image_index = i * n + j
            if kernel.shape[-1] == 1:
                axes[i, j].imshow(kernel[i, j, :, 0], cmap='gray')
            else:
                axes[i, j].imshow(kernel[i, j, :, :])
            axes[i, j].axis('off')

    st.pyplot(fig)



if __name__ == "__main__":
    st.title("CNN Kernel Visualization")

    # Interactive model selection
    model_name = st.selectbox(
        "Select a model:",
        ["AlexNet", "EfficientNet_B0", "MNASNet0_5", "MobileNet_V3_Small", "RegNet_X_1_6GF"]
    )

    main(model_name)
