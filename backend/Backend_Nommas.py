import os
import logging
import json
import base64
import cv2
import torch
import numpy as np
import threading
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
from train import run

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

detection_data = []

# Initialize SIFT detector
sift = cv2.SIFT_create(nfeatures=1500)

bbox = None
selected = False
roi_keypoints = None
roi_descriptors = None
roi = None
cap = cv2.VideoCapture(0)

# BFMatcher with default params
bf = cv2.BFMatcher()

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

def generate_sift_frames():
    global bbox, selected, roi_keypoints, roi_descriptors, roi, cap

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        raise RuntimeError("Error: Could not open webcam.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        img = frame.copy()

        if selected and roi_descriptors is not None:
            frame_keypoints, frame_descriptors = sift.detectAndCompute(frame, None)

            if frame_descriptors is not None and len(frame_descriptors) > 0:
                matches = bf.knnMatch(roi_descriptors, frame_descriptors, k=2)

                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

                if len(good_matches) > 5:
                    src_pts = np.float32([roi_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([frame_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                    if M is not None and M.shape == (3, 3):
                        h, w = roi.shape[:2]
                        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                        dst = cv2.perspectiveTransform(pts, M)

                        x, y, w, h = cv2.boundingRect(dst)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    else:
                        print("Homography matrix is not valid.")
                else:
                    print("Not enough good matches.")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def sift_video_feed():
    return Response(generate_sift_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def process_bbox(data):
    global bbox, selected, roi_keypoints, roi_descriptors, roi, cap

    x1 = int(data['x1'])
    y1 = int(data['y1'])
    x2 = int(data['x2'])
    y2 = int(data['y2'])
    bbox = (x1, y1, x2, y2)
    selected = True

    ret, frame = cap.read()
    if ret:
        roi = frame[y1:y2, x1:x2]
        roi_keypoints, roi_descriptors = sift.detectAndCompute(roi, None)

    detection_geometry = {
        "x": (x1 / frame.shape[1]) * 100,
        "y": (y1 / frame.shape[0]) * 100,
        "width": ((x2 - x1) / frame.shape[1]) * 100,
        "height": ((y2 - y1) / frame.shape[0]) * 100,
    }

    return detection_geometry

@app.route('/set_bbox', methods=['POST'])
def set_bbox():
    data = request.json
    detection_geometry = process_bbox(data)
    return jsonify(success=True, detectionGeometry=detection_geometry)

@app.route('/save-annotations', methods=['POST'])
def save_annotations():
    try:
        data = request.json
        annotations = data['annotations']
        dataset_folder = data['datasetFolder']
        train_percent = data['trainPercent']
        val_percent = data['valPercent']
        test_percent = data['testPercent']
        tags = data['tags']

        base_dir = os.path.join(dataset_folder)
        os.makedirs(os.path.join(base_dir, 'train', 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'train', 'labels'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'valid', 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'valid', 'labels'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'test', 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'test', 'labels'), exist_ok=True)

        train, val, test = [], [], []

        for item in annotations:
            label = item['label']
            if label not in tags, continue

            rand = os.urandom(1)[0] / 255.0
            if rand < float(train_percent) / 100:
                train.append(item)
            elif rand < (float(train_percent) + float(val_percent)) / 100:
                val.append(item)
            else:
                test.append(item)

        def save_annotations(data, type):
            for anno in data:
                image_id = anno['imageId']
                label = anno['label']
                try:
                    label_index = tags.index(label)
                except ValueError:
                    continue
                x_center = anno['x_center'] / 100
                y_center = anno['y_center'] / 100
                bbox_width = anno['bbox_width'] / 100
                bbox_height = anno['bbox_height'] / 100
                image_data = anno['imageData']

                image_filename = os.path.basename(image_id) + '.png'
                label_filename = os.path.basename(image_id) + '.txt'

                label_file = os.path.join(base_dir, type, 'labels', label_filename)
                with open(label_file, 'a') as f:
                    annotation_str = f"{label_index} {x_center} {y_center} {bbox_width} {bbox_height}\n"
                    f.write(annotation_str)

                image_file = os.path.join(base_dir, type, 'images', image_filename)
                with open(image_file, 'wb') as f:
                    try:
                        if image_data.startswith('data:image'):
                            f.write(base64.b64decode(image_data.split(',')[1]))
                    except Exception as e:
                        continue

        save_annotations(train, 'train')
        save_annotations(val, 'valid')
        save_annotations(test, 'test')

        data_yaml = f"""
train: {os.path.join(base_dir, 'train', 'images').replace('\\\\', '/')}
val: {os.path.join(base_dir, 'valid', 'images').replace('\\\\', '/')}
nc: {len(tags)}
names: {json.dumps(tags)}
"""

        data_yaml_path = os.path.join(base_dir, 'data.yaml')
        with open(data_yaml_path, 'w') as f:
            f.write(data_yaml)

        return jsonify(message='Annotations saved and training data prepared successfully.', data_yaml_path=data_yaml_path)
    except Exception as e:
        return jsonify(error=f'Failed to parse request data. {str(e)}'), 500

@app.route('/start-training', methods=['GET'])
def start_training():
    dataset_folder = request.args.get('dataset_folder')
    data_yaml_path = os.path.join(dataset_folder, 'data.yaml')
    logger.info(f"Starting training with data: {data_yaml_path}")

    def run_training(data_path):
        try:
            run(data=data_path, weights='yolov5s.pt', epochs=1, batch_size=2, imgsz=640, callbacks=None)
            return True, "Training completed successfully."
        except Exception as e:
            logger.error(str(e))
            return False, str(e)

    success, output = run_training(data_yaml_path)
    
    if success:
        return jsonify(message='Training started successfully.')
    else:
        return jsonify(error='Training failed.', details=output), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    threading.Thread(target=generate_frames, args=("runs/train/exp3/weights/best.pt", 0, (640, 640), 0.25, 0.45, 1000, "", None, False, 3, False, False, False, False)).start()
    threading.Thread(target=generate_sift_frames).start()
    app.run(host="0.0.0.0", port=port)
import os
import logging
import json
import base64
import cv2
import torch
import numpy as np
import threading
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
from train import run

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

detection_data = []

# Initialize SIFT detector
sift = cv2.SIFT_create(nfeatures=1500)

bbox = None
selected = False
roi_keypoints = None
roi_descriptors = None
roi = None
cap = cv2.VideoCapture(0)

# BFMatcher with default params
bf = cv2.BFMatcher()

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

def generate_sift_frames():
    global bbox, selected, roi_keypoints, roi_descriptors, roi, cap

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        raise RuntimeError("Error: Could not open webcam.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        img = frame.copy()

        if selected and roi_descriptors is not None:
            frame_keypoints, frame_descriptors = sift.detectAndCompute(frame, None)

            if frame_descriptors is not None and len(frame_descriptors) > 0:
                matches = bf.knnMatch(roi_descriptors, frame_descriptors, k=2)

                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

                if len(good_matches) > 5:
                    src_pts = np.float32([roi_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([frame_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                    if M is not None and M.shape == (3, 3):
                        h, w = roi.shape[:2]
                        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                        dst = cv2.perspectiveTransform(pts, M)

                        x, y, w, h = cv2.boundingRect(dst)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    else:
                        print("Homography matrix is not valid.")
                else:
                    print("Not enough good matches.")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def sift_video_feed():
    return Response(generate_sift_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def process_bbox(data):
    global bbox, selected, roi_keypoints, roi_descriptors, roi, cap

    x1 = int(data['x1'])
    y1 = int(data['y1'])
    x2 = int(data['x2'])
    y2 = int(data['y2'])
    bbox = (x1, y1, x2, y2)
    selected = True

    ret, frame = cap.read()
    if ret:
        roi = frame[y1:y2, x1:x2]
        roi_keypoints, roi_descriptors = sift.detectAndCompute(roi, None)

    detection_geometry = {
        "x": (x1 / frame.shape[1]) * 100,
        "y": (y1 / frame.shape[0]) * 100,
        "width": ((x2 - x1) / frame.shape[1]) * 100,
        "height": ((y2 - y1) / frame.shape[0]) * 100,
    }

    return detection_geometry

@app.route('/set_bbox', methods=['POST'])
def set_bbox():
    data = request.json
    detection_geometry = process_bbox(data)
    return jsonify(success=True, detectionGeometry=detection_geometry)

@app.route('/save-annotations', methods=['POST'])
def save_annotations():
    try:
        data = request.json
        annotations = data['annotations']
        dataset_folder = data['datasetFolder']
        train_percent = data['trainPercent']
        val_percent = data['valPercent']
        test_percent = data['testPercent']
        tags = data['tags']

        base_dir = os.path.join(dataset_folder)
        os.makedirs(os.path.join(base_dir, 'train', 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'train', 'labels'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'valid', 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'valid', 'labels'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'test', 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'test', 'labels'), exist_ok=True)

        train, val, test = [], [], []

        for item in annotations:
            label = item['label']
            if label not in tags:
                continue

            rand = os.urandom(1)[0] / 255.0
            if rand < float(train_percent) / 100:
                train.append(item)
            elif rand < (float(train_percent) + float(val_percent)) / 100:
                val.append(item)
            else:
                test.append(item)

        def save_annotations(data, type):
            for anno in data:
                image_id = anno['imageId']
                label = anno['label']
                try:
                    label_index = tags.index(label)
                except ValueError:
                    continue
                x_center = anno['x_center'] / 100
                y_center = anno['y_center'] / 100
                bbox_width = anno['bbox_width'] / 100
                bbox_height = anno['bbox_height'] / 100
                image_data = anno['imageData']

                image_filename = os.path.basename(image_id) + '.png'
                label_filename = os.path.basename(image_id) + '.txt'

                label_file = os.path.join(base_dir, type, 'labels', label_filename)
                with open(label_file, 'a') as f:
                    annotation_str = f"{label_index} {x_center} {y_center} {bbox_width} {bbox_height}\n"
                    f.write(annotation_str)

                image_file = os.path.join(base_dir, type, 'images', image_filename)
                with open(image_file, 'wb') as f:
                    try:
                        if image_data.startswith('data:image'):
                            f.write(base64.b64decode(image_data.split(',')[1]))
                    except Exception as e:
                        continue

        save_annotations(train, 'train')
        save_annotations(val, 'valid')
        save_annotations(test, 'test')

        data_yaml = f"""
train: {os.path.join(base_dir, 'train', 'images').replace('\\\\', '/')}
val: {os.path.join(base_dir, 'valid', 'images').replace('\\\\', '/')}
nc: {len(tags)}
names: {json.dumps(tags)}
"""

        data_yaml_path = os.path.join(base_dir, 'data.yaml')
        with open(data_yaml_path, 'w') as f:
            f.write(data_yaml)

        return jsonify(message='Annotations saved and training data prepared successfully.', data_yaml_path=data_yaml_path)
    except Exception as e:
        return jsonify(error=f'Failed to parse request data. {str(e)}'), 500

@app.route('/start-training', methods=['GET'])
def start_training():
    dataset_folder = request.args.get('dataset_folder')
    data_yaml_path = os.path.join(dataset_folder, 'data.yaml')
    logger.info(f"Starting training with data: {data_yaml_path}")

    def run_training(data_path):
        try:
            run(data=data_path, weights='yolov5s.pt', epochs=1, batch_size=2, imgsz=640, callbacks=None)
            return True, "Training completed successfully."
        except Exception as e:
            logger.error(str(e))
            return False, str(e)

    success, output = run_training(data_yaml_path)
    
    if success:
        return jsonify(message='Training started successfully.')
    else:
        return jsonify(error='Training failed.', details=output), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    threading.Thread(target=generate_frames, args=("runs/train/exp3/weights/best.pt", 0, (640, 640), 0.25, 0.45, 1000, "", None, False, 3, False, False, False, False)).start()
    threading.Thread(target=generate_sift_frames).start()
    app.run(host="0.0.0.0", port=port)
