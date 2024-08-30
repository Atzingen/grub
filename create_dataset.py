import numpy as np
import cv2

cap = cv2.VideoCapture('video.mp4')
# cap = cv2.VideoCapture('denoise_video.avi')
background_sub = cv2.createBackgroundSubtractorMOG2(history=5000, 
                                                    varThreshold=40, 
                                                    detectShadows=False)
# background_sub = cv2.createBackgroundSubtractorKNN()

ksize = 3
min_area = 25
max_area = 120

ret, frame = cap.read()
frame_idx = 0
for i in range(5000):
    _, _ = cap.read()
    if i % 50 == 0:
        cv2.imwrite(f'annotated/images/frame_{frame_idx}.jpg', frame)
        with open(f'annotated/labels/frame_{frame_idx}.txt', 'w') as f:
            f.write('0 0.575800 0.135000 0.010000 0.013500\n')
            f.write('0 0.565000 0.904000 0.012000 0.013500\n')
            f.write('0 0.187000 0.055000 0.005708 0.014363\n')
        frame_idx += 1
    frame_idx += 1

while True:
    for i in range(150):
        _, _ = cap.read()
        frame_idx += 1
    ret, frame = cap.read()
    frame_idx += 1

    with open(f'annotated/labels/frame_{frame_idx}.txt', 'w') as f:
        print(frame_idx)
        if not ret:
            break
        fgMask = background_sub.apply(frame)
        kernel = np.ones((ksize, ksize), np.uint8)
        cleaned_image = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > min_area and cv2.contourArea(contour) < max_area:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                x, y, w, h = cv2.boundingRect(contour)
                x_center = (x + w / 2) / frame.shape[1]
                y_center = (y + h / 2) / frame.shape[0]
                width = w / frame.shape[1]
                height = h / frame.shape[0]
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        f.write('0 0.575800 0.135000 0.010000 0.013500\n')
        f.write('0 0.565000 0.904000 0.012000 0.013500\n')
        f.write('0 0.187000 0.055000 0.005708 0.014363\n')
        cv2.imwrite(f'annotated/images/frame_{frame_idx}.jpg', frame)
        combined_image = np.hstack((frame, cv2.cvtColor(cleaned_image, cv2.COLOR_GRAY2BGR)))
        combined_image = cv2.resize(combined_image, (1280, 640))
        cv2.imshow('Frame', combined_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()