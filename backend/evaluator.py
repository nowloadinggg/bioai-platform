

import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Track and calculate performance metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Clear all tracked data"""
        self.inference_times = []
        self.detection_counts = []
        self.confidence_scores = []
        logger.info("📊 Metrics reset")
    
    def add_result(self, result: Dict):
        """
        Add detection result for tracking
        
        Args:
            result: Detection result from detector
        """
        self.inference_times.append(result['inference_time_ms'])
        self.detection_counts.append(result['count'])
        
        # Track confidence scores
        for det in result['detections']:
            self.confidence_scores.append(det['confidence'])
    
    def get_metrics(self) -> Dict:
        """
        Calculate aggregated metrics
        
        Returns:
            {
                'total_images': int,
                'total_detections': int,
                'avg_detections_per_image': float,
                'avg_confidence': float,
                'avg_inference_ms': float,
                'fps': float,
                'min_inference_ms': float,
                'max_inference_ms': float
            }
        """
        if not self.inference_times:
            return {
                'total_images': 0,
                'total_detections': 0,
                'avg_detections_per_image': 0,
                'avg_confidence': 0,
                'avg_inference_ms': 0,
                'fps': 0,
                'min_inference_ms': 0,
                'max_inference_ms': 0
            }
        
        avg_inference = np.mean(self.inference_times)
        fps = 1000 / avg_inference if avg_inference > 0 else 0
        avg_confidence = np.mean(self.confidence_scores) if self.confidence_scores else 0
        
        return {
            'total_images': len(self.inference_times),
            'total_detections': sum(self.detection_counts),
            'avg_detections_per_image': round(np.mean(self.detection_counts), 2),
            'avg_confidence': round(avg_confidence, 3),
            'avg_inference_ms': round(avg_inference, 2),
            'fps': round(fps, 2),
            'min_inference_ms': round(min(self.inference_times), 2),
            'max_inference_ms': round(max(self.inference_times), 2)
        }