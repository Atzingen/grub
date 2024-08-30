from ultralytics import YOLO

def train_model():
    model = YOLO('yolov8m.pt')

    results = model.train(
        data='datasets/grobs-5/data.yaml',  # Path to your data.yaml file
        epochs=120,                # Number of epochs
        imgsz=1080,                # Image size
        batch=-1,                  # Batch size
        # lr0=0.001,                 # Learning rate
        name='small_object_model', # Name of the training session
        augment=True,              # Enable data augmentation (default is True)
    )
    metrics = model.val()
    print(metrics)
    # results = model('path/to/image.jpg')
    # results.show()

if __name__ == '__main__':
    train_model()