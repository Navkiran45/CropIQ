import torch
import cv2
import numpy as np
from ultralytics import YOLO
from dotenv import load_dotenv
import os
import h5py
import tqdm

# Load environment variables
load_dotenv()
PROCESSED_DATA_PATH = os.getenv("PROCESSED_DATA_PATH")
FEATURES_OUTPUT_PATH = os.getenv("FEATURES_OUTPUT_PATH", os.path.join(PROCESSED_DATA_PATH, "features"))

# Ensure features output directory exists
os.makedirs(FEATURES_OUTPUT_PATH, exist_ok=True)

# Load YOLOv9 model
model = YOLO("yolov9c.pt")

def extract_features_batch(img_paths):
    """
    Extract features for a batch of images
    
    Args:
        img_paths (list): List of image file paths
    
    Returns:
        torch.Tensor: Batch of extracted features
    """
    # Preprocess images
    imgs = [cv2.imread(path) for path in img_paths]
    imgs = [cv2.resize(img, (640, 640)) for img in imgs]  # Resize to YOLO input size
    
    # Convert to tensor and normalize
    imgs_tensor = torch.tensor(imgs).float().permute(0, 3, 1, 2) / 255.0
    
    # Extract features using forward hook
    features = []
    
    def hook_function(module, input, output):
        features.append(output)
    
    # Register hook on one of the intermediate layers
    hook = model.model.model[9].register_forward_hook(hook_function)
    
    # Forward pass
    with torch.no_grad():
        model(imgs_tensor)
    
    # Remove the hook
    hook.remove()
    
    return features[0]

def extract_dataset_features(data_path, batch_size=32):
    """
    Extract features for entire dataset
    
    Args:
        data_path (str): Path to image dataset
        batch_size (int): Number of images to process in each batch
    """
    # Get all image paths
    image_paths = [
        os.path.join(data_path, img) for img in os.listdir(data_path) 
        if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ]
    
    # Prepare HDF5 file for storing features
    h5_path = os.path.join(FEATURES_OUTPUT_PATH, 'dataset_features.h5')
    with h5py.File(h5_path, 'w') as h5f:
        # Create datasets to store features and file names
        feature_dataset = h5f.create_dataset(
            'features', 
            shape=(len(image_paths), *extract_features_batch([image_paths[0]]).shape[1:]), 
            dtype='float32'
        )
        filename_dataset = h5f.create_dataset(
            'filenames', 
            shape=(len(image_paths),), 
            dtype=h5py.special_dtype(vlen=str)
        )
        
        # Process images in batches
        for i in tqdm.tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i+batch_size]
            batch_features = extract_features_batch(batch_paths)
            
            # Store features and filenames
            feature_dataset[i:i+len(batch_features)] = batch_features.numpy()
            filename_dataset[i:i+len(batch_features)] = [os.path.basename(path) for path in batch_paths]
    
    print(f"Features extracted and saved to {h5_path}")
    return h5_path

if __name__ == "__main__":
    # Extract features for entire dataset
    features_file = extract_dataset_features(PROCESSED_DATA_PATH)
    
    # Optionally, verify the extracted features
    with h5py.File(features_file, 'r') as h5f:
        print("Total images processed:", len(h5f['filenames']))
        print("Feature shape:", h5f['features'].shape)
        
        # Example: Print first few filenames and feature vectors
        print("\nFirst 5 filenames:")
        print(h5f['filenames'][:5])
        print("\nFirst feature vector:")
        print(h5f['features'][0])