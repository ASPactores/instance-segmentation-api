import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from core.config import settings

class YOLOService:
    def __init__(self):
        self.model = None
        
    def load_model(self):
        """Load the YOLO model."""
        self.model = YOLO(settings.MODEL_PATH)
        
    def predict(self, image_array):
        """Run prediction on an image."""
        if self.model is None:
            self.load_model()
            
        # Convert image to OpenCV format for YOLO (RGB to BGR)
        opencv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Run inference
        return self.model(opencv_image, conf=settings.CONFIDENCE_THRESHOLD)

yolo_service = YOLOService()