import time
import os
import base64
from io import BytesIO
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from PIL import Image, ImageDraw

# ---------------- FLASK APP ----------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------------- MODEL (Load once at startup) ----------------
print("Loading YOLO model...")
model = YOLO("best.pt")
CONF = 0.3
print("Model loaded successfully!")

# ---------------- DETECTION LOGIC (Preserved from original) ----------------
def detect_people(image):
    """
    Core ML detection logic - UNCHANGED from original implementation
    Detects people in the provided image and returns annotated results
    """
    start = time.time()

    if image is None:
        return None, [], 0, "LOW", 0

    img = image.convert("RGB")
    results = model.predict(img, conf=CONF, verbose=False)

    draw = ImageDraw.Draw(img)
    logs = []
    count = 0
    detections = []

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls].lower()
            conf = float(box.conf[0])

            if label not in ["person", "human"]:
                continue

            count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw bounding box with cyan color
            pulse = int(2 + conf * 4)
            draw.rectangle([x1, y1, x2, y2], outline="#00fff7", width=pulse)
            draw.text((x1, y1 - 14), f"HUMAN {conf:.2f}", fill="#00fff7")

            logs.append(f"TARGET LOCKED | CONF={conf:.2f}")
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": round(conf, 2),
                "label": "HUMAN"
            })

    if count == 0:
        logs.append("AREA CLEAR")

    fps = 1 / max(time.time() - start, 0.001)

    # Threat level calculation
    threat = "LOW"
    if count >= 3:
        threat = "HIGH"
    elif count > 0:
        threat = "MEDIUM"

    return img, logs, count, threat, fps, detections

# ---------------- FLASK ROUTES ----------------

@app.route('/')
def index():
    """Serve the main UI"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for people detection
    Accepts: Image file upload or camera blob
    Returns: JSON with detection results
    """
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({
                "status": "error",
                "message": "No image file provided"
            }), 400

        file = request.files['image']
        
        # Allow empty filename for camera captures (blobs)
        if file.filename == '' and not file:
            return jsonify({
                "status": "error",
                "message": "Empty file"
            }), 400

        # Read and process the image
        try:
            image = Image.open(file.stream)
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Invalid image format: {str(e)}"
            }), 400
        
        # Run detection using existing ML logic
        annotated_img, logs, count, threat, fps, detections = detect_people(image)

        # Convert annotated image to base64 for response
        buffered = BytesIO()
        annotated_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Return clean JSON response
        return jsonify({
            "status": "success",
            "people_count": count,
            "threat_level": threat,
            "fps": round(fps, 1),
            "logs": logs,
            "detections": detections,
            "annotated_image": f"data:image/png;base64,{img_str}"
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# ---------------- RUN SERVER ----------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=True)
