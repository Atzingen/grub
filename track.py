import cv2
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO('models/best.pt')

# Track objects in a video
# results = model.track(source='video.mp4', 
                    #   conf=0.1,
                    #   show=True,
                    #   save=True)
                    #   tracker='botsort.yaml')

# results = model.predict(source='videos/video.mp4', 
#                       conf=0.1,
#                       show=True,
#                       save=True)


cap = cv2.VideoCapture("videos/video.mp4")

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # results = model.track(frame, 
        #                       tracker='custon_tracker.yaml',
        #                       persist=True)
        results = model.predict(frame, 
                              conf=0.1)
        annotated_frame = results[0].plot()
        cv2.imshow("Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()