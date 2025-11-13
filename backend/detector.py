

import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List
import time
import logging

logger = logging.getLogger(__name__)


class OocyteDetector:
    """YOLO detector for oocytes with performance tracking"""
    
    def __init__(self, model_path: str = '../yolo11n.pt'):
        """Initialize YOLO model"""
        try:
            logger.info(f"Loading model: {model_path}")
            self.model = YOLO(model_path)
            self.model_name = model_path.replace('.pt', '').split('/')[-1]
            logger.info(f"✅ Model {self.model_name} loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def detect(self, image: np.ndarray, conf: float = 0.25, iou: float = 0.45) -> Dict:
        """
        Detect oocytes in image
        
        Args:
            image: Input image (numpy array)
            conf: Confidence threshold
            iou: IoU threshold for NMS
            
        Returns:
            {
                'detections': [...],
                'count': int,
                'inference_time_ms': float,
                'image_shape': [h, w]
            }
        """
        start_time = time.time()
        
        # Run YOLO inference
        results = self.model(image, conf=conf, iou=iou, verbose=False)
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Extract detections
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].item()
                
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(confidence),
                    'area': float((x2 - x1) * (y2 - y1))
                })
        
        return {
            'detections': detections,
            'count': len(detections),
            'inference_time_ms': round(inference_time, 2),
            'image_shape': list(image.shape[:2])
        }


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Convert bytes to OpenCV image (RGB)"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)