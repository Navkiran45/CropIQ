import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CONFIG = {
    "IMG_SIZE": 640,
    "BATCH_SIZE": 8,
    "EPOCHS": 50,
    "MODEL_PATH": "models/yolov9c.pt",
    "DATA_PATHS": {
        "RAW": os.getenv("RAW_DATA_PATH"),
        "PROCESSED": os.getenv("PROCESSED_DATA_PATH"),
        "LABELS": os.getenv("LABELS_PATH")
    }
}
