import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from ultralytics import YOLO
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
PROCESSED_DATA_PATH = os.getenv("PROCESSED_DATA_PATH")

# Load YOLOv9 model
model = YOLO("yolov9c.pt")

def visualize_features(img_path):
    # Read and preprocess image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (640, 640))
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor
    input_tensor = torch.from_numpy(rgb_img.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
    
    # Extract features using hooks
    features = []
    def hook_fn(module, input, output):
        features.append(output.detach().cpu())
    
    # Register hook on layer 9 (you can change this to extract from different layers)
    hook = model.model.model[9].register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        model(input_tensor)
    hook.remove()
    
    # Get feature maps
    feature_maps = features[0][0].numpy()
    
    # Visualize feature maps
    plt.figure(figsize=(15, 15))
    
    # Plot original image
    plt.subplot(5, 4, 1)
    plt.imshow(rgb_img)
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot first 15 feature channels
    for i in range(15):
        plt.subplot(5, 4, i+2)
        plt.imshow(feature_maps[i], cmap='viridis')
        plt.title(f'Feature Channel {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Create and display average activation map
    avg_activation = np.mean(feature_maps, axis=0)
    plt.figure(figsize=(10, 10))
    plt.imshow(avg_activation, cmap='hot')
    plt.title('Average Feature Activation')
    plt.colorbar()
    plt.axis('off')
    plt.show()
    
    # Overlay average activation on original image
    avg_activation = cv2.resize(avg_activation, (640, 640))
    avg_activation = (avg_activation - avg_activation.min()) / (avg_activation.max() - avg_activation.min())
    heatmap = np.uint8(255 * avg_activation)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title('Feature Activation Overlay')
    plt.axis('off')
    plt.show()

# Get the first image from your processed directory
sample_image = os.path.join(PROCESSED_DATA_PATH, os.listdir(PROCESSED_DATA_PATH)[0])
print("Using image:", sample_image)
visualize_features(sample_image)