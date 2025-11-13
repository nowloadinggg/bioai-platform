from flask import Flask, request, jsonify, send_from_directory
from inference_sdk import InferenceHTTPClient
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
import cv2

# ----------------------------
# Configuration
# ----------------------------
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

WORKSPACE = "meo-d8oog"       # Your Roboflow workspace
WORKFLOW_ID = "detect-count-and-visualize"  # Your Roboflow workflow ID
MODEL_ID = "meo-d8oog/1"  # Your Roboflow model ID (workspace/version)
API_KEY = "rf_sowvghtKgpcpTYwdyZnwol2E9Rg2"  # Your Roboflow API key

# ----------------------------
# Initialize Flask
# ----------------------------
app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ----------------------------
# Initialize Roboflow client
# ----------------------------
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)

# ----------------------------
# Helper Functions
# ----------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def draw_bounding_boxes(filepath, predictions):
    """Draw bounding boxes on the image using predictions."""
    image = cv2.imread(filepath)
    if image is None:
        return None

    for pred in predictions:
        x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
        class_name = pred["class"]
        conf = pred["confidence"]
        
        # Color: Green for good, Red for bad
        color = (0, 255, 0) if class_name == "good" else (0, 0, 255)
        
        # Calculate box corners
        x1, y1 = x - w // 2, y - h // 2
        x2, y2 = x + w // 2, y + h // 2
        
        # Draw rectangle with thicker lines
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        
        # Create label with background
        label = f"{class_name} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw filled rectangle as background for text
        cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
        
        # Put text on top of background
        cv2.putText(image, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness)

    annotated_filename = f"annotated_{os.path.basename(filepath)}"
    annotated_path = os.path.join(UPLOAD_FOLDER, annotated_filename)
    cv2.imwrite(annotated_path, image)
    return annotated_filename


# ----------------------------
# Routes
# ----------------------------
@app.route("/api/analyze", methods=["POST"])
def analyze_images():
    if "images" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist("images")
    results = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            try:
                # Send image to Roboflow using infer method
                result = client.infer(filepath, model_id=MODEL_ID)

                # Extract predictions
                image_url = None
                predictions = result.get("predictions", []) if isinstance(result, dict) else []

                # Generate visualization manually
                if predictions:
                    annotated_filename = draw_bounding_boxes(filepath, predictions)
                    if annotated_filename:
                        image_url = f"http://127.0.0.1:8080/uploads/{annotated_filename}"

                # Format result to match frontend expectations
                formatted_result = {
                    "results": [{
                        "predictions": predictions,
                        "visualization": image_url
                    }]
                }

                results.append({
                    "filename": filename,
                    "image_url": image_url,
                    "result": formatted_result
                })

            except Exception as e:
                results.append({
                    "filename": filename,
                    "error": str(e)
                })

    return jsonify({"status": "success", "results": results})


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    """Serve uploaded and annotated images."""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "BioAI Backend Running"})


# ----------------------------
# Run App
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
