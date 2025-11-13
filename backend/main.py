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
    """Draw bounding boxes on the image using predictions (if Roboflow doesn't return visualization)."""
    image = cv2.imread(filepath)
    if image is None:
        return None

    for pred in predictions:
        x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
        class_name = pred["class"]
        conf = pred["confidence"]
        color = (0, 255, 0) if class_name == "good" else (0, 0, 255)
        cv2.rectangle(image, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color, 2)
        cv2.putText(image, f"{class_name} ({conf:.2f})", (x - w // 2, y - h // 2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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
                # Send image to Roboflow
                result = client.run_workflow(
                    workspace_name=WORKSPACE,
                    workflow_id=WORKFLOW_ID,
                    images={"image": filepath},
                    use_cache=True
                )

                # Extract visualized image if available
                image_url = None
                predictions = []
                if isinstance(result, dict):
                    rf_results = result.get("results", [])
                    if rf_results:
                        predictions = rf_results[0].get("predictions", [])
                        if "visualization" in rf_results[0]:
                            image_url = rf_results[0]["visualization"]

                # If Roboflow didn't return visualization, generate it manually
                if not image_url and predictions:
                    annotated_filename = draw_bounding_boxes(filepath, predictions)
                    if annotated_filename:
                        image_url = f"http://127.0.0.1:8080/uploads/{annotated_filename}"

                results.append({
                    "filename": filename,
                    "image_url": image_url,
                    "result": result
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
