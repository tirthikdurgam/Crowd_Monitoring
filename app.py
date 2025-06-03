from flask import Flask, request, jsonify, send_from_directory, send_file
import os
import cv2
from ultralytics import YOLO
import uuid

app = Flask(__name__, static_folder='static')

# Folder setup
upload_folder = 'uploads'
output_folder = 'outputs'
os.makedirs(upload_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# Load YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# Serve the index.html from the 'static' folder
@app.route('/')
def index():
    return send_file('static/index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400

    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    filename = f"{uuid.uuid4()}.mp4"
    filepath = os.path.join(upload_folder, filename)
    video.save(filepath)

    return jsonify({'message': 'Upload successful', 'filename': filename}), 200

@app.route('/process', methods=['POST'])
def process_video():
    data = request.json
    if not data or 'filename' not in data:
        return jsonify({'error': 'Filename is required'}), 400

    filename = data['filename']
    input_path = os.path.join(upload_folder, filename)
    if not os.path.exists(input_path):
        return jsonify({'error': 'File not found'}), 404

    output_filename = f"processed_{filename}"
    output_path = os.path.join(output_folder, output_filename)

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25  # fallback FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        count = 0
        for box in results.boxes:
            cls = int(box.cls)
            if model.names[cls] == 'person':
                count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


        cv2.putText(frame, f'People Count: {count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        out.write(frame)

    cap.release()
    out.release()

    return jsonify({'message': 'Processing complete',
                    'output_url': f'/download/{output_filename}'}), 200

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory(output_folder, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
