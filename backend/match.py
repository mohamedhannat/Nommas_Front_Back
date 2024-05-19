# import cv2
# import numpy as np

# # Initialize the webcam
# cap = cv2.VideoCapture(1)

# # Initialize SIFT detector
# sift = cv2.SIFT_create()

# # Define the callback function for drawing a rectangle
# def draw_rectangle(event, x, y, flags, param):
#     global bbox, drawing, ix, iy, selected, roi_descriptors, roi_keypoints, roi

#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         ix, iy = x, y

#     elif event == cv2.EVENT_MOUSEMOVE:
#         if drawing:
#             img2 = img.copy()
#             cv2.rectangle(img2, (ix, iy), (x, y), (0, 255, 0), 2)
#             cv2.imshow('frame', img2)

#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False
#         bbox = (ix, iy, x, y)
#         cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
#         cv2.imshow('frame', img)
#         # Extract features from the selected ROI
#         roi = img[iy:y, ix:x]
#         roi_keypoints, roi_descriptors = sift.detectAndCompute(roi, None)
#         selected = True
#         print(f"Selected ROI Keypoints: {len(roi_keypoints)}")

# # Initialize variables
# bbox = None
# drawing = False
# ix, iy = -1, -1
# selected = False
# roi_keypoints = None
# roi_descriptors = None
# roi = None

# cv2.namedWindow('frame')
# cv2.setMouseCallback('frame', draw_rectangle)

# # BFMatcher with default params
# bf = cv2.BFMatcher()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     img = frame.copy()

#     if selected and roi_descriptors is not None:
#         # Extract features from the current frame
#         frame_keypoints, frame_descriptors = sift.detectAndCompute(frame, None)
        
#         if frame_descriptors is not None and len(frame_descriptors) > 0:
#             # Match descriptors
#             matches = bf.knnMatch(roi_descriptors, frame_descriptors, k=2)

#             # Apply ratio test
#             good_matches = []
#             for m, n in matches:
#                 if m.distance < 0.75 * n.distance:
#                     good_matches.append(m)

#             # Draw matches
#             matching_result = cv2.drawMatches(img, roi_keypoints, frame, frame_keypoints, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#             #cv2.imshow('matching', matching_result)

#             if len(good_matches) > 5:  # Ensure there are enough good matches
#                 src_pts = np.float32([roi_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#                 dst_pts = np.float32([frame_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
#                 # Find homography matrix
#                 M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#                 matches_mask = mask.ravel().tolist()

#                 h, w = roi.shape[:2]
#                 pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
#                 dst = cv2.perspectiveTransform(pts, M)

#                 # Get bounding box coordinates from the transformed points
#                 x, y, w, h = cv2.boundingRect(dst)
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
#             else:
#                 matches_mask = None

#         cv2.imshow('frame', frame)

#     else:
#         cv2.imshow('frame', frame)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(1)

# Initialize SIFT detector with increased number of features
sift = cv2.SIFT_create(nfeatures=1500)

# Define the callback function for drawing a rectangle
def draw_rectangle(event, x, y, flags, param):
    global bbox, drawing, ix, iy, selected, roi_descriptors, roi_keypoints, roi

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
        # Extract features from the selected ROI
        roi = img[iy:y, ix:x]
        roi_keypoints, roi_descriptors = sift.detectAndCompute(roi, None)
        selected = True
        print(f"Selected ROI Keypoints: {len(roi_keypoints)}")

# Initialize variables
bbox = None
drawing = False
ix, iy = -1, -1
selected = False
roi_keypoints = None
roi_descriptors = None
roi = None

cv2.namedWindow('frame')
cv2.setMouseCallback('frame', draw_rectangle)

# BFMatcher with default params
bf = cv2.BFMatcher()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    img = frame.copy()

    if selected and roi_descriptors is not None:
        # Extract features from the current frame
        frame_keypoints, frame_descriptors = sift.detectAndCompute(frame, None)
        
        if frame_descriptors is not None and len(frame_descriptors) > 0:
            # Match descriptors
            matches = bf.knnMatch(roi_descriptors, frame_descriptors, k=2)

            # Apply ratio test with a more lenient threshold
            good_matches = []
            for m, n in matches:
                if m.distance < 0.85 * n.distance:
                    good_matches.append(m)

            # Draw matches
            matching_result = cv2.drawMatches(img, roi_keypoints, frame, frame_keypoints, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow('matching', matching_result)

            if len(good_matches) > 5:  # Ensure there are enough good matches
                src_pts = np.float32([roi_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([frame_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Find homography matrix
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if M is not None and M.shape == (3, 3):  # Ensure the homography matrix is valid
                    h, w = roi.shape[:2]
                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)

                    # Get bounding box coordinates from the transformed points
                    x, y, w, h = cv2.boundingRect(dst)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                else:
                    print("Homography matrix is not valid.")
            else:
                print("Not enough good matches.")

        cv2.imshow('frame', frame)

    else:
        cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
