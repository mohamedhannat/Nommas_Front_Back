import os
import cv2
import numpy as np
import base64
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

sift = cv2.SIFT_create(nfeatures=1500)
bf = cv2.BFMatcher()

bbox = None
selected = False
roi_keypoints = None
roi_descriptors = None
roi = None

def process_sift_frame(frame):
    global bbox, selected, roi_keypoints, roi_descriptors, roi

    if selected and roi_descriptors is not None:
        frame_keypoints, frame_descriptors = sift.detectAndCompute(frame, None)

        if frame_descriptors is not None and len(frame_descriptors) > 0:
            matches = bf.knnMatch(roi_descriptors, frame_descriptors, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

            if len(good_matches) > 5:
                src_pts = np.float32([roi_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([frame_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if M is not None and M.shape == (3, 3):
                    h, w = roi.shape[:2]
                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)

                    x, y, w, h = cv2.boundingRect(dst)
                    x = max(0, min(x, frame.shape[1]))
                    y = max(0, min(y, frame.shape[0]))
                    w = max(0, min(w, frame.shape[1] - x))
                    h = max(0, min(h, frame.shape[0] - y))

                    detection_geometry = {
                        "x": (x / frame.shape[1]) * 100,
                        "y": (y / frame.shape[0]) * 100,
                        "width": (w / frame.shape[1]) * 100,
                        "height": (h / frame.shape[0]) * 100,
                    }
                    return detection_geometry
                else:
                    logger.debug("Homography matrix is not valid.")
            else:
                logger.debug("Not enough good matches.")
    return None

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global bbox, selected, roi_keypoints, roi_descriptors, roi

    data = request.json
    image_data = data['image']

    # Decode image
    try:
        image_data = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"Error decoding image: {e}")
        return jsonify(error="Error decoding image"), 400

    if frame is None or frame.size == 0:
        logger.error("Empty frame received")
        return jsonify(error="Empty frame"), 400

    detection_geometry = process_sift_frame(frame)

    return jsonify(success=bool(detection_geometry), detectionGeometry=detection_geometry) if detection_geometry else jsonify(success=False)

@app.route('/set_bbox', methods=['POST'])
def set_bbox():
    global bbox, selected, roi_keypoints, roi_descriptors, roi

    data = request.json
    image_data = data['image']
    bbox = data['bbox']

    # Decode image
    try:
        image_data = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"Error decoding image: {e}")
        return jsonify(error="Error decoding image"), 400

    if frame is None or frame.size == 0:
        logger.error("Empty frame received")
        return jsonify(error="Empty frame"), 400

    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
    if x1 >= x2 or y1 >= y2:
        logger.error("Invalid bounding box coordinates")
        return jsonify(error="Invalid bounding box coordinates"), 400

    x1 = max(0, min(x1, frame.shape[1]))
    y1 = max(0, min(y1, frame.shape[0]))
    x2 = max(0, min(x2, frame.shape[1]))
    y2 = max(0, min(y2, frame.shape[0]))

    roi = frame[y1:y2, x1:x2]
    if roi is None or roi.size == 0:
        logger.error("Empty ROI")
        return jsonify(error="Empty ROI"), 400

    roi_keypoints, roi_descriptors = sift.detectAndCompute(roi, None)
    if roi_keypoints is None or roi_descriptors is None:
        logger.error("No keypoints or descriptors found in ROI")
        return jsonify(error="No keypoints or descriptors found in ROI"), 400

    selected = True

    detection_geometry = {
        "x": (x1 / frame.shape[1]) * 100,
        "y": (y1 / frame.shape[0]) * 100,
        "width": ((x2 - x1) / frame.shape[1]) * 100,
        "height": ((y2 - y1) / frame.shape[0]) * 100,
    }

    return jsonify(success=True, detectionGeometry=detection_geometry)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
