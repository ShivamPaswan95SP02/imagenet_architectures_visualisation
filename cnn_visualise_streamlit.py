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


    # Plot kernel grid for the selected layer
    # if hasattr(model, layer_name):
    #     layer = getattr(model, layer_name)
    #     if hasattr(layer, 'weight'):
    #         plot_kernel_grid(layer.weight)
    #     else:
    #         st.write("Selected layer does not have weights to visualize.")
    # else:
    #     st.write("Layer not found in the model.")

    # Load an image from a local path
    # img = Image.open(image_path)

    # # Preprocess the image
    # preprocess = weights.transforms()
    # batch = preprocess(img).unsqueeze(0)

    # # Perform inference
    # prediction = model(batch).squeeze(0).softmax(0)
    # class_id = prediction.argmax().item()
    # score = prediction[class_id].item()
    # category_name = weights.meta["categories"][class_id]
    # st.write(f"{category_name}: {100 * score:.1f}%")

if __name__ == "__main__":
    st.title("CNN Kernel Visualization")

    # Interactive model selection
    model_name = st.selectbox(
        "Select a model:",
        ["AlexNet", "EfficientNet_B0", "MNASNet0_5", "MobileNet_V3_Small", "RegNet_X_1_6GF"]
    )

    main(model_name)



