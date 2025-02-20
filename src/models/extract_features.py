import torch
import cv2
import numpy as np
from ultralytics import YOLO
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
PROCESSED_DATA_PATH = os.getenv("PROCESSED_DATA_PATH")

# Load YOLOv9 model
model = YOLO("yolov9c.pt")

def extract_features(img_path):
    # Read and preprocess image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (640, 640))  # Resize to YOLO input size
    
    # Convert to tensor and normalize
    img_tensor = torch.tensor(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    
    # Extract features using forward hook
    features = []
    
    def hook_function(module, input, output):
        features.append(output)
    
    # Register hook on one of the intermediate layers
    # You can adjust which layer to extract features from
    hook = model.model.model[9].register_forward_hook(hook_function)  # Example: layer 9
    
    # Forward pass
    with torch.no_grad():
        model(img_tensor)
    
    # Remove the hook
    hook.remove()
    
    return features[0]  # Return the extracted features

# Example usage
sample_image = os.path.join(PROCESSED_DATA_PATH, os.listdir(PROCESSED_DATA_PATH)[0])
features = extract_features(sample_image)
print("Extracted Feature Shape:", features.shape)