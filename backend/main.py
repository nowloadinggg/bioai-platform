from flask import Flask, request, jsonify, send_from_directory
from inference_sdk import InferenceHTTPClient
from werkzeug.utils import secure_filename
from flask_cors import CORS
from dotenv import load_dotenv
import os
import cv2

# ----------------------------
# Load config.env
# ----------------------------
load_dotenv("config.env")

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
ALLOWED_EXTENSIONS = set(os.getenv("ALLOWED_EXTENSIONS", "png,jpg,jpeg").split(","))

WORKSPACE = os.getenv("WORKSPACE")
MODEL_ID = os.getenv("MODEL_ID")
WORKFLOW_ID = os.getenv("WORKFLOW_ID")
API_KEY = os.getenv("API_KEY")

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
    image = cv2.imread(filepath)
    if image is None:
        return None

    for pred in predictions:
        x, y = int(pred["x"]), int(pred["y"])
        w, h = int(pred["width"]), int(pred["height"])
        class_name = pred["class"]
        conf = pred["confidence"]

        # Color: Green for good, Red for bad
        color = (0, 255, 0) if class_name == "good" else (0, 0, 255)

        x1, y1 = x - w // 2, y - h // 2
        x2, y2 = x + w // 2, y + h // 2

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

        label = f"{class_name} {conf:.2f}"

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

        cv2.rectangle(image, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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
                result = client.infer(filepath, model_id=MODEL_ID)
                predictions = result.get("predictions", []) if isinstance(result, dict) else []

                image_url = None
                if predictions:
                    annotated_filename = draw_bounding_boxes(filepath, predictions)
                    if annotated_filename:
                        image_url = f"http://127.0.0.1:8080/uploads/{annotated_filename}"

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
                results.append({"filename": filename, "error": str(e)})

    return jsonify({"status": "success", "results": results})


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "BioAI Backend Running"})

# ----------------------------
# Run App
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, debug=True)