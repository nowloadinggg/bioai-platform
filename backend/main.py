from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from yolo_model import OocyteYOLO
from grading import OocyteGrader
import uuid
from datetime import datetime
import os

app = FastAPI(
    title="BioAI Oocyte Grading API",
    description="AI-powered oocyte detection and grading system",
    version="1.0.0"
)

# CORS - Allow all origins for demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
yolo_model = OocyteYOLO(
    detect_model_path='models/yolo11n.pt',
    segment_model_path='models/yolo11n-seg.pt'
)
grader = OocyteGrader()

# Create uploads directory
os.makedirs('uploads', exist_ok=True)

@app.get("/")
async def root():
    return {
        "message": "BioAI Oocyte Grading API",
        "version": "1.0.0",
        "status": "running"
    }

@app.post("/api/v1/analysis/submit")
async def analyze_oocyte(
    file: UploadFile = File(...),
    confidence: float = Form(0.25)
):
    """
    Main endpoint for oocyte analysis
    
    Expected by frontend:
    - analysis_id: unique ID
    - data: array of oocytes with grading
    """
    try:
        # Generate analysis ID
        analysis_id = str(uuid.uuid4())[:8]
        
        # Read image
        image_bytes = await file.read()
        
        # Step 1: Detect oocytes
        detections = yolo_model.detect_oocytes(image_bytes, conf=confidence)
        
        if not detections:
            return JSONResponse(
                status_code=200,
                content={
                    "analysis_id": analysis_id,
                    "data": [],
                    "message": "No oocytes detected in the image"
                }
            )
        
        # Step 2: Segment and extract features
        segments = yolo_model.segment_oocytes(image_bytes, conf=confidence)
        
        # Step 3: Grade oocytes
        graded_oocytes = grader.grade_batch(segments)
        
        # Step 4: Create annotated image (optional)
        # annotated_image = yolo_model.create_annotated_image(image_bytes, detections)
        
        # Return response matching frontend expectation
        return JSONResponse(
            status_code=200,
            content={
                "analysis_id": analysis_id,
                "timestamp": datetime.now().isoformat(),
                "total_count": len(graded_oocytes),
                "data": graded_oocytes,
                # "annotated_image": annotated_image  # Optional
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": True,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)