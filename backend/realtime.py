import os
import cv2
import torch
from pathlib import Path
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
from models.common import DetectMultiBackend
from utils.general import (
    check_img_size,
    non_max_suppression,
    scale_boxes,
)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode
import threading

app = Flask(__name__)
CORS(app)

detection_data = []

def generate_frames(weights, source, imgsz, conf_thres, iou_thres, max_det, device, classes, agnostic_nms, line_thickness, hide_labels, hide_conf, half, dnn):
    # Convert parameters to correct types
    imgsz = tuple(map(int, imgsz))
    conf_thres = float(conf_thres)
    iou_thres = float(iou_thres)
    max_det = int(max_det)
    line_thickness = int(line_thickness)
    hide_labels = bool(hide_labels)
    hide_conf = bool(hide_conf)
    half = bool(half)
    dnn = bool(dnn)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Initialize webcam
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open webcam {source}")
        return

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Preprocess image
        img = torch.from_numpy(frame).to(model.device)
        img = img.permute(2, 0, 1)  # convert to [channels, height, width]
        img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img.unsqueeze(0)  # expand for batch dim

        # Inference
        pred = model(img)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        detections = []
        for i, det in enumerate(pred):  # per image
            annotator = Annotator(frame, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to frame size
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()

                # Collect results
                for *xyxy, conf, cls in reversed(det):
                    if conf > conf_thres:
                        label = None if hide_labels else (names[int(cls)] if hide_conf else f"{names[int(cls)]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(int(cls), True))
                        detections.append({
                            "label": label,
                            "x": xyxy[0].item() / frame.shape[1] * 100,
                            "y": xyxy[1].item() / frame.shape[0] * 100,
                            "width": (xyxy[2].item() - xyxy[0].item()) / frame.shape[1] * 100,
                            "height": (xyxy[3].item() - xyxy[1].item()) / frame.shape[0] * 100,
                        })

            global detection_data
            detection_data = detections

            # Stream results
            frame = annotator.result()
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/yolo_video_feed')
def video_feed():
    # Fetch parameters from the last start_detection call
    global run_params
    return Response(generate_frames(**run_params), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global run_params
    run_params = request.json
    return jsonify(success=True)

@app.route('/detection_data', methods=['GET'])
def get_detection_data():
    return jsonify(detectionGeometry=detection_data)

@app.route('/upload_model', methods=['POST'])
def upload_model():
    if 'file' not in request.files:
        return jsonify(error='No file part'), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify(error='No selected file'), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join('models', filename)
        file.save(filepath)
        return jsonify(path=filepath), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
