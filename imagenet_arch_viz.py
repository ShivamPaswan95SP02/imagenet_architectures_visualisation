import torch
import pdb
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

# Plotting ----
def plot_model_summary(model):
    # Plot a simple summary of the model
    plt.figure(figsize=(10, 6))
    plt.title("Model Summary")
    plt.text(0.1, 0.5, str(model), fontsize=10, va='top')
    plt.axis('off')
    plt.show()

def load_image_from_path(image_path):
    # Load an image from a local path
    img = Image.open(image_path)
    return img


def plot_kernel_grid(kernel) :

    kernel = kernel.detach().numpy()

    m, n, s, _ = kernel.shape

    fig, axes = plt.subplots(m, n, figsize=(s * n,s * m))
    for i in range(m):
        for j in range(n):
            image_index = i * n + j
            if kernel.shape[-1] == 1:
                axes[i, j].imshow(kernel[i,j, :, 0], cmap='gray')
            else:
                axes[i, j].imshow(kernel[i,j, :, :])
            axes[i, j].axis('off')
    plt.show()

def main(model_name, image_path):
    # Load the model
    model, weights = load_model(model_name)

    # List layers
    layers = list_layers(model)
    print("Model Layers:")
    for layer in layers:
        print(layer)

    # Plot model summary
    # plot_model_summary(model)

    plot_kernel_grid(model.features[0].weight) 

    # Load an image from a local path
    img = load_image_from_path(image_path)

    # Preprocess the image
    preprocess = weights.transforms()
    batch = preprocess(img).unsqueeze(0)

    # Perform inference
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {100 * score:.1f}%")

if __name__ == "__main__":
    # Interactive model selection
    print("Available models: AlexNet, EfficientNet_B0, MNASNet0_5, MobileNet_V3_Small, RegNet_X_1_6GF")
    # model_name = input("Enter the model name you want to use: ")
    # image_path = input("Enter the path to your image: ")

    model_name = 'AlexNet'
    image_path = r'C:\Users\trainee\Desktop\Capture.png'

    # Run the main function with the selected model and image path
    main(model_name, image_path)
