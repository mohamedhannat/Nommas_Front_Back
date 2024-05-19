# import cv2
# import numpy as np
# from flask import Flask, Response

# app = Flask(__name__)

# def generate_frames():
#     # Use DirectShow backend for capturing video
#     cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#     if not cap.isOpened():
#         raise RuntimeError("Error: Could not open webcam.")

#     while True:
#         success, frame = cap.read()
#         if not success:
#             break
#         else:
#             # Convert the frame to HSV
#             hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#             # Define the range for yellow color in HSV
#             lower_yellow = np.array([20, 100, 100])
#             upper_yellow = np.array([30, 255, 255])

#             # Threshold the HSV image to get only yellow colors
#             mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

#             # Find contours
#             contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#             # Draw bounding boxes around detected contours
#             for contour in contours:
#                 if cv2.contourArea(contour) > 500:  # Filter small contours
#                     x, y, w, h = cv2.boundingRect(contour)
#                     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#             # Encode the frame in JPEG format
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()

#             # Yield the output frame in byte format
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/video_feed')
# def video_feed():
#     # Video streaming route.
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=5000)


import cv2
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(1)

# Define the callback function for drawing a rectangle
def draw_rectangle(event, x, y, flags, param):
    global bbox, drawing, ix, iy, selected, color_lower, color_upper

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img2 = img.copy()
            cv2.rectangle(img2, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('frame', img2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bbox = (ix, iy, x, y)
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow('frame', img)
        # Calculate color range
        color_lower, color_upper = get_color_range(bbox, img)
        selected = True
        print(f"Color range selected: Lower={color_lower}, Upper={color_upper}")

# Function to get the color range from the selected bbox
def get_color_range(bbox, frame):
    x1, y1, x2, y2 = bbox
    # Ensure the coordinates are within the frame bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    roi = frame[y1:y2, x1:x2]
    cv2.imshow('roi', roi)  # Display the ROI for verification
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Calculate the mean and standard deviation of the hue, saturation, and value channels
    hue_mean, hue_std = np.mean(hsv_roi[:, :, 0]), np.std(hsv_roi[:, :, 0])
    sat_mean, sat_std = np.mean(hsv_roi[:, :, 1]), np.std(hsv_roi[:, :, 1])
    val_mean, val_std = np.mean(hsv_roi[:, :, 2]), np.std(hsv_roi[:, :, 2])
    lower_bound = np.array([max(0, hue_mean - hue_std), max(50, sat_mean - sat_std), max(50, val_mean - val_std)])
    upper_bound = np.array([min(179, hue_mean + hue_std), min(255, sat_mean + sat_std), min(255, val_mean + val_std)])
    return lower_bound, upper_bound

# Initialize variables
bbox = None
drawing = False
ix, iy = -1, -1
selected = False
color_lower = np.array([0, 0, 0])
color_upper = np.array([0, 0, 0])

cv2.namedWindow('frame')
cv2.setMouseCallback('frame', draw_rectangle)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    img = frame.copy()

    if selected:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, color_lower, color_upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Detect contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 800:  # Adjust the area threshold as needed
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('mask', mask)
        cv2.imshow('result', result)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


