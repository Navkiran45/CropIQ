import os
import cv2
import numpy as np
import albumentations as A
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
RAW_DATA_PATH = os.getenv("RAW_DATA_PATH")
PROCESSED_DATA_PATH = os.getenv("PROCESSED_DATA_PATH")

# Create processed directory if not exists
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

# Define augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussNoise(p=0.2)
])

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (640, 640))  # Resize to 640x640
    return img  # Return without normalization

# Process images
for img_file in os.listdir(RAW_DATA_PATH):
    img_path = os.path.join(RAW_DATA_PATH, img_file)
    img = preprocess_image(img_path)
    
    # Apply augmentations before normalization
    aug_img = transform(image=img)["image"]
    
    # Normalize after augmentations
    aug_img = aug_img.astype(np.float32) / 255.0
    
    # Convert back to uint8 for saving
    save_img = (aug_img * 255).astype(np.uint8)
    save_path = os.path.join(PROCESSED_DATA_PATH, img_file)
    cv2.imwrite(save_path, save_img)

print("Preprocessing complete. Processed images saved to:", PROCESSED_DATA_PATH)