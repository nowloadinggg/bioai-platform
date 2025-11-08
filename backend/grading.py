from typing import Dict, List

class OocyteGrader:
    """
    WHO-based oocyte grading system
    """
    
    def __init__(self):
        # Grading thresholds (simplified)
        self.thresholds = {
            'circularity_min': 0.7,      # Good circularity
            'zp_thickness_min': 10.0,     # Minimum ZP thickness
            'zp_thickness_max': 25.0,     # Maximum ZP thickness
            'area_min': 5000,             # Minimum pixel area
        }
    
    def grade_oocyte(self, segment: Dict) -> Dict:
        """
        Grade a single oocyte based on morphological features
        
        Returns:
            Dict with grade (A/B/C) and score (0-100)
        """
        score = 0
        max_score = 100
        
        # 1. Circularity (40 points)
        circularity = segment.get('circularity', 0)
        if circularity >= 0.9:
            score += 40
        elif circularity >= 0.8:
            score += 30
        elif circularity >= 0.7:
            score += 20
        else:
            score += 10
        
        # 2. Zona Pellucida Thickness (30 points)
        zp_thickness = segment.get('zp_thickness', 0)
        if self.thresholds['zp_thickness_min'] <= zp_thickness <= self.thresholds['zp_thickness_max']:
            score += 30
        elif zp_thickness < self.thresholds['zp_thickness_min']:
            score += 15
        else:
            score += 10
        
        # 3. Size/Area (20 points)
        area = segment.get('pixel_area', 0)
        if area >= self.thresholds['area_min']:
            score += 20
        else:
            score += int((area / self.thresholds['area_min']) * 20)
        
        # 4. Confidence (10 points)
        confidence = segment.get('confidence', 0)
        score += int(confidence * 10)
        
        # Normalize to 100
        score = min(score, max_score)
        
        # Determine grade
        if score >= 80:
            grade = "Tốt (Good/A)"
            grade_en = "A"
        elif score >= 60:
            grade = "Trung bình (Fair/B)"
            grade_en = "B"
        else:
            grade = "Kém (Poor/C)"
            grade_en = "C"
        
        return {
            'grade': grade,
            'grade_en': grade_en,
            'score': score,
            'details': {
                'circularity': circularity,
                'zp_thickness': zp_thickness,
                'area': area,
                'confidence': confidence
            }
        }
    
    def grade_batch(self, segments: List[Dict]) -> List[Dict]:
        """Grade multiple oocytes"""
        graded = []
        
        for segment in segments:
            grading = self.grade_oocyte(segment)
            
            graded.append({
                'oocyte_id': segment['oocyte_id'],
                'bbox': segment['bbox'],
                'pixel_area': segment['pixel_area'],
                'confidence': segment['confidence'],
                'who_grading': grading
            })
        
        return graded