import cv2
import numpy as np

# Open the video file
cap = cv2.VideoCapture('video.mp4')

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('denoise_video.avi', fourcc, fps, (frame_width, frame_height))

# Read and process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Apply denoising
    denoised_frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

    # Write the frame to the output video
    out.write(denoised_frame)

    print(ret)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when the job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
