import os
from typing import Tuple, List
import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

class DataLoader:
    def __init__(self):
        self.raw_data_path = os.getenv('RAW_DATA_PATH')
        self.processed_data_path = os.getenv('PROCESSED_DATA_PATH')
        self.labels_path = os.getenv('LABELS_PATH')
        
    def load_dataset(self) -> Tuple[List[np.ndarray], List[str]]:
        """
        Load images and their corresponding labels
        """
        images = []
        labels = []
        # Implement data loading logic here
        return images, labels
    
    def split_dataset(self, test_size: float = 0.2):
        """
        Split dataset into train and test sets
        """
        pass 