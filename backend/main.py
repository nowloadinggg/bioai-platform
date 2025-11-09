from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from yolo_detector import get_detector
from grading import grader
import uuid
from datetime import datetime
import time

app = FastAPI(
    title="BioAI Oocyte Grading API",
    description="AI-powered oocyte detection and grading using YOLOv11",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Load YOLO models on startup"""
    print("🚀 Starting BioAI Backend...")
    get_detector()  # Load models
    print("✅ Backend ready!")

@app.get("/")
async def root():
    return {
        "message": "BioAI Oocyte Grading API",
        "version": "1.0.0",
        "status": "running",
        "model": "YOLOv11"
    }

@app.post("/api/v1/analysis/submit")
async def analyze_oocyte(
    file: UploadFile = File(...),
    confidence: float = Form(0.25)
):
    """
    Main endpoint: Analyze oocyte image
    
    Returns:
        JSON với analysis_id, data (graded oocytes)
    """
    try:
        print(f"📸 Analyzing image: {file.filename}")

        # Generate analysis ID
        analysis_id = str(uuid.uuid4())[:8]

        timings = {}
        t0 = time.perf_counter()

        # Read image
        t_read_start = time.perf_counter()
        image_bytes = await file.read()
        t_read_end = time.perf_counter()
        timings['read_seconds'] = round(t_read_end - t_read_start, 4)

        # Get detector (should be loaded at startup)
        t_detector_start = time.perf_counter()
        detector = get_detector()
        t_detector_end = time.perf_counter()
        timings['get_detector_seconds'] = round(t_detector_end - t_detector_start, 4)

        # Step 1: Detect và segment oocytes
        print("🔍 Running YOLO detection...")
        t_infer_start = time.perf_counter()
        oocytes = detector.detect_and_segment(image_bytes, conf=confidence)
        t_infer_end = time.perf_counter()
        timings['inference_seconds'] = round(t_infer_end - t_infer_start, 4)

        if not oocytes:
            print("⚠️ No oocytes detected")
            total_time = round(time.perf_counter() - t0, 4)
            timings['total_seconds'] = total_time
            return JSONResponse(
                status_code=200,
                content={
                    "analysis_id": analysis_id,
                    "timestamp": datetime.now().isoformat(),
                    "total_count": 0,
                    "data": [],
                    "message": "No oocytes detected. Try adjusting confidence threshold.",
                    "timings": timings
                }
            )

        print(f"✅ Detected {len(oocytes)} oocytes")

        # Step 2: Grade oocytes
        print("📊 Grading oocytes...")
        t_grade_start = time.perf_counter()
        graded_oocytes = grader.grade_batch(oocytes)
        t_grade_end = time.perf_counter()
        timings['grading_seconds'] = round(t_grade_end - t_grade_start, 4)

        # Step 3: Create annotated image (optional)
        # annotated_image = detector.create_annotated_image(image_bytes, oocytes)

        total_time = round(time.perf_counter() - t0, 4)
        timings['total_seconds'] = total_time

        print(f"✅ Analysis complete: {analysis_id} (took {total_time}s)")

        # Return response
        return JSONResponse(
            status_code=200,
            content={
                "analysis_id": analysis_id,
                "timestamp": datetime.now().isoformat(),
                "total_count": len(graded_oocytes),
                "data": graded_oocytes,
                "timings": timings
                # "annotated_image": annotated_image  # Uncomment nếu muốn trả về ảnh
            }
        )
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/api/v1/health")
async def health_check():
    """Health check"""
    try:
        detector = get_detector()
        return {
            "status": "healthy",
            "models_loaded": True,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)