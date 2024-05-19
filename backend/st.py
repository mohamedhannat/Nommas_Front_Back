import cv2
import numpy as np
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
import threading

app = Flask(__name__)
CORS(app)





# Initialize SIFT detector
sift = cv2.SIFT_create(nfeatures=1500)

bbox = None
selected = False
roi_keypoints = None
roi_descriptors = None
roi = None
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# BFMatcher with default params
bf = cv2.BFMatcher()

def generate_frames():
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
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

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

    # Assuming the detected bounding box is the same as the selected one for simplicity
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




if __name__ == '__main__':
    threading.Thread(target=generate_frames).start()
    app.run(host='0.0.0.0', port=5000)
