import cv2
import numpy as np
from typing import Tuple, List

class CNNPreprocessor:
    def __init__(self, input_size: Tuple[int, int] = (416, 416)):
        self.input_size = input_size
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for CNN input
        """
        # Resize image
        resized = cv2.resize(image, self.input_size)
        
        # Normalize
        normalized = resized / 255.0
        
        # Add batch dimension
        preprocessed = np.expand_dims(normalized, axis=0)
        
        return preprocessed
    
    def augment_data(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Apply data augmentation techniques
        """
        augmented = []
        # Add your augmentation techniques here
        return augmented 