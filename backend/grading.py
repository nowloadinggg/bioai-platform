from typing import Dict, List

class OocyteGrader:
    """WHO-based grading system"""
    
    def __init__(self):
        self.thresholds = {
            'circularity_excellent': 0.85,
            'circularity_good': 0.75,
            'circularity_fair': 0.65,
            'zp_min': 10.0,
            'zp_max': 25.0,
            'area_min': 5000,
        }
    
    def grade_oocyte(self, oocyte: Dict) -> Dict:
        """
        Grade 1 oocyte dựa trên morphological features
        
        Returns:
            Dict với grade (A/B/C) và score (0-100)
        """
        score = 0
        print(f"\n🔍 Grading oocyte {oocyte.get('oocyte_id')}:")
        print(f"  - Input features: {oocyte}")
        
        # 1. Circularity (40 points)
        circularity = oocyte.get('circularity', 0)
        if circularity >= self.thresholds['circularity_excellent']:
            score += 40
        elif circularity >= self.thresholds['circularity_good']:
            score += 30
        elif circularity >= self.thresholds['circularity_fair']:
            score += 20
        else:
            score += 10
        
        # 2. ZP Thickness (30 points)
        zp = oocyte.get('zp_thickness', 0)
        if self.thresholds['zp_min'] <= zp <= self.thresholds['zp_max']:
            score += 30
        elif zp < self.thresholds['zp_min']:
            score += 15
        else:
            score += 10
        
        # 3. Size (20 points)
        area = oocyte.get('pixel_area', 0)
        if area >= self.thresholds['area_min']:
            score += 20
        else:
            score += int((area / self.thresholds['area_min']) * 20)
        
        # 4. Confidence (10 points)
        conf = oocyte.get('confidence', 0)
        score += int(conf * 10)
        
        # Print score components
        print(f"  - Score breakdown:")
        print(f"    * Circularity (max 40): {circularity} -> {score - (int(conf * 10) + int((area / self.thresholds['area_min']) * 20) + (30 if self.thresholds['zp_min'] <= zp <= self.thresholds['zp_max'] else 15 if zp < self.thresholds['zp_min'] else 10))}")
        print(f"    * ZP Thickness (max 30): {zp} -> {30 if self.thresholds['zp_min'] <= zp <= self.thresholds['zp_max'] else 15 if zp < self.thresholds['zp_min'] else 10}")
        print(f"    * Size (max 20): {area}/{self.thresholds['area_min']} -> {20 if area >= self.thresholds['area_min'] else int((area / self.thresholds['area_min']) * 20)}")
        print(f"    * Confidence (max 10): {conf} -> {int(conf * 10)}")
        
        # Normalize
        score = min(score, 100)
        print(f"  - Final score: {score}")
        
        # Determine grade
        if score >= 80:
            grade = "Tốt (Good/A)"
        elif score >= 60:
            grade = "Trung bình (Fair/B)"
        else:
            grade = "Kém (Poor/C)"
        print(f"  - Final grade: {grade}")
        
        return {
            'grade': grade,
            'score': score
        }
    
    def grade_batch(self, oocytes: List[Dict]) -> List[Dict]:
        """Grade multiple oocytes"""
        graded = []
        
        for oocyte in oocytes:
            grading = self.grade_oocyte(oocyte)
            
            graded.append({
                'oocyte_id': oocyte['oocyte_id'],
                'bbox': oocyte['bbox'],
                'pixel_area': oocyte['pixel_area'],
                'confidence': oocyte['confidence'],
                'who_grading': grading
            })
        
        return graded

# Global instance
grader = OocyteGrader()