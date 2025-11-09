import cv2
import numpy as np
from ultralytics import YOLO
import base64
import os
from typing import List, Dict

class OocyteDetector:
    def __init__(self):
        """
        Initialize YOLO models
        Models sẽ tự động download nếu chưa có!
        """
        try:
            print("🔄 Loading YOLO models...")
            print(f"Current working directory: {os.getcwd()}")
            
            # Get absolute paths to models
            model_dir = os.path.abspath(os.path.dirname(__file__))
            detect_model_path = os.path.join(model_dir, 'yolo11n.pt')
            segment_model_path = os.path.join(model_dir, 'yolo11n-seg.pt')
            
            print(f"Loading detection model from: {detect_model_path}")
            print(f"Loading segmentation model from: {segment_model_path}")
            
            if not os.path.exists(detect_model_path):
                raise FileNotFoundError(f"Detection model not found at {detect_model_path}")
            if not os.path.exists(segment_model_path):
                raise FileNotFoundError(f"Segmentation model not found at {segment_model_path}")
            
            self.detect_model = YOLO(detect_model_path)
            self.segment_model = YOLO(segment_model_path)
            
            print("✅ YOLO models loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading YOLO models: {str(e)}")
            raise
    
    def detect_and_segment(self, image_bytes: bytes, conf: float = 0.25) -> List[Dict]:
        """
        Detect và segment oocytes từ image
        
        Returns:
            List of oocytes với bbox, mask, features
        """
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            
            if image is None:
                raise ValueError("Cannot decode image")
                
            # Convert to RGB if needed
            if len(image.shape) == 2:  # grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 3:  # BGR
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run segmentation (bao gồm cả detection)
            print(f"🔍 Running segmentation with confidence threshold: {conf}")
            results = self.segment_model(image, conf=conf, verbose=True)  # Enable verbose for debugging
            print(f"✅ Segmentation completed. Shape of image: {image.shape}")
            
            oocytes = []
            
            for r in results:
                if r.boxes is None:
                    continue
                
                boxes = r.boxes
                masks = r.masks.data.cpu().numpy() if r.masks is not None else None
                
                for idx, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = box.conf[0].item()
                    
                    # Calculate features
                    bbox_area = (x2 - x1) * (y2 - y1)
                    
                    # Extract mask features nếu có
                    if masks is not None and idx < len(masks):
                        mask = masks[idx]
                        pixel_area = int(np.sum(mask))
                        circularity = self._calculate_circularity(mask)
                        zp_thickness = self._estimate_zp_thickness(mask)
                    else:
                        pixel_area = int(bbox_area)
                        circularity = 0.8  # Default
                        zp_thickness = 15.0  # Default
                    
                    oocytes.append({
                        'oocyte_id': idx + 1,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': round(confidence, 3),
                        'pixel_area': pixel_area,
                        'circularity': round(circularity, 3),
                        'zp_thickness': round(zp_thickness, 2)
                    })
            
            return oocytes
        
        except Exception as e:
            print(f"❌ Detection error: {str(e)}")
            raise
    
    def _calculate_circularity(self, mask: np.ndarray) -> float:
        """Calculate circularity (4πA/P²)"""
        try:
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask_uint8, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return 0.8
            
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter == 0:
                return 0.8
            
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            return min(circularity, 1.0)
        
        except:
            return 0.8
    
    def _estimate_zp_thickness(self, mask: np.ndarray) -> float:
        """Estimate zona pellucida thickness"""
        try:
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask_uint8, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return 15.0
            
            contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(contour)
            
            if M["m00"] == 0:
                return 15.0
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            distances = [
                np.sqrt((pt[0][0]-cx)**2 + (pt[0][1]-cy)**2) 
                for pt in contour
            ]
            
            return np.mean(distances) * 0.1
        
        except:
            return 15.0
    
    def create_annotated_image(self, image_bytes: bytes, oocytes: List[Dict]) -> str:
        """
        Tạo ảnh có annotations (bounding boxes + labels)
        
        Returns:
            Base64 encoded image
        """
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Draw annotations
            for oocyte in oocytes:
                x1, y1, x2, y2 = oocyte['bbox']
                oocyte_id = oocyte['oocyte_id']
                conf = oocyte['confidence']
                
                # Color based on confidence
                if conf > 0.8:
                    color = (0, 255, 0)  # Green
                elif conf > 0.6:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 0, 255)  # Red
                
                # Draw rectangle
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"#{oocyte_id} ({conf:.2f})"
                cv2.putText(
                    image, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )
            
            # Encode to base64
            _, buffer = cv2.imencode('.jpg', image)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return f'data:image/jpeg;base64,{img_base64}'
        
        except Exception as e:
            print(f"❌ Annotation error: {str(e)}")
            return ""

# Global instance
detector = None

def get_detector():
    """Singleton pattern - chỉ load model 1 lần"""
    global detector
    if detector is None:
        detector = OocyteDetector()
    return detector