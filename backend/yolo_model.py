import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict
import base64

class OocyteYOLO:
    def __init__(self, 
                 detect_model_path='models/yolo11n.pt',
                 segment_model_path='models/yolo11n-seg.pt'):
        """Initialize YOLO models"""
        self.detect_model = YOLO(detect_model_path)
        self.segment_model = YOLO(segment_model_path)
        
    def detect_oocytes(self, image_bytes: bytes, conf: float = 0.25) -> List[Dict]:
        """
        Detect oocytes in image
        
        Returns:
            List of detections with bbox, confidence, etc.
        """
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run detection
        results = self.detect_model(image, conf=conf)
        
        # Extract detections
        detections = []
        for r in results:
            boxes = r.boxes
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                
                detections.append({
                    'oocyte_id': idx + 1,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': round(confidence, 3),
                    'center': [int((x1+x2)/2), int((y1+y2)/2)]
                })
        
        return detections
    
    def segment_oocytes(self, image_bytes: bytes, conf: float = 0.25) -> List[Dict]:
        """
        Segment oocytes and extract morphological features
        
        Returns:
            List of segments with masks, areas, etc.
        """
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run segmentation
        results = self.segment_model(image, conf=conf)
        
        # Extract segments
        segments = []
        for r in results:
            if r.masks is not None:
                masks = r.masks.data.cpu().numpy()
                boxes = r.boxes
                
                for idx, (mask, box) in enumerate(zip(masks, boxes)):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = box.conf[0].item()
                    
                    # Calculate morphological features
                    mask_area = np.sum(mask)
                    bbox_area = (x2 - x1) * (y2 - y1)
                    circularity = self._calculate_circularity(mask)
                    
                    # Extract zona pellucida thickness (simplified)
                    zp_thickness = self._estimate_zp_thickness(mask)
                    
                    segments.append({
                        'oocyte_id': idx + 1,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': round(confidence, 3),
                        'pixel_area': int(mask_area),
                        'bbox_area': int(bbox_area),
                        'circularity': round(circularity, 3),
                        'zp_thickness': round(zp_thickness, 2),
                        'mask': mask  # For further processing
                    })
        
        return segments
    
    def _calculate_circularity(self, mask: np.ndarray) -> float:
        """Calculate circularity of mask (4πA/P²)"""
        # Convert to uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(
            mask_uint8, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return 0.0
        
        # Get largest contour
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return 0.0
        
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        return min(circularity, 1.0)  # Cap at 1.0
    
    def _estimate_zp_thickness(self, mask: np.ndarray) -> float:
        """Estimate zona pellucida thickness (simplified)"""
        # This is a simplified version
        # In production, you'd use erosion/dilation to detect ZP
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(
            mask_uint8, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return 0.0
        
        contour = max(contours, key=cv2.contourArea)
        
        # Simple approximation: distance from centroid to edge
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return 0.0
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Calculate average distance
        distances = [np.sqrt((pt[0][0]-cx)**2 + (pt[0][1]-cy)**2) 
                    for pt in contour]
        
        return np.mean(distances) * 0.1  # Scale factor (arbitrary)
    
    def create_annotated_image(self, 
                               image_bytes: bytes, 
                               detections: List[Dict]) -> str:
        """
        Create annotated image with bounding boxes
        
        Returns:
            Base64 encoded image
        """
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Draw bounding boxes
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            oocyte_id = det['oocyte_id']
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Oocyte #{oocyte_id}"
            cv2.putText(image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Encode to base64
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return f'data:image/jpeg;base64,{img_base64}'