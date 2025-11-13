"""
Configuration Settings
Author: @nowloadinggg
Date: 2025-11-09 11:35:14 UTC
"""

# API Settings
API_HOST = "0.0.0.0"
API_PORT = 8080  

# Model Paths
YOLO_DETECTION_MODEL = '../yolo11n.pt'
YOLO_SEGMENTATION_MODEL = '../yolo11n-seg.pt'

# Detection Parameters
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.45

# CORS Origins
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]