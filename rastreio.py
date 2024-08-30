import numpy as np
import cv2

cap = cv2.VideoCapture('videos/video.mp4')
# cap = cv2.VideoCapture('denoise_video.avi')
background_sub = cv2.createBackgroundSubtractorMOG2(history=5000, 
                                                    varThreshold=40, 
                                                    detectShadows=False)
# background_sub = cv2.createBackgroundSubtractorKNN()
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Frame',1280, 640)

def nothing(x):
    pass

cv2.createTrackbar('Kernel Size', 'Frame', 3, 20, nothing)
cv2.createTrackbar('Min Area', 'Frame', 25, 50, nothing)
cv2.createTrackbar('Max Area', 'Frame', 120, 200, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    ksize = cv2.getTrackbarPos('Kernel Size', 'Frame')
    min_area = cv2.getTrackbarPos('Min Area', 'Frame')
    max_area = cv2.getTrackbarPos('Max Area', 'Frame')

    fgMask = background_sub.apply(frame)
    kernel = np.ones((ksize, ksize), np.uint8)
    cleaned_image = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > min_area and cv2.contourArea(contour) < max_area:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(frame, center, radius + 4, (0, 255, 0), 4)  # Draw green circl
    combined_image = np.hstack((frame, cv2.cvtColor(cleaned_image, cv2.COLOR_GRAY2BGR)))
    combined_image = cv2.resize(combined_image, (1280, 640))
    cv2.imshow('Frame', combined_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()