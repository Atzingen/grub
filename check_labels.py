import cv2
import matplotlib.pyplot as plt
import os

def plot_bounding_boxes(image_path, label_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with open(label_path, 'r') as file:
        lines = file.readlines()

    height, width, _ = image.shape
    for line in lines:
        class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.split())
        x_center, y_center = x_center * width, y_center * height
        bbox_width, bbox_height = bbox_width * width, bbox_height * height
        x_min = int(x_center - bbox_width / 2)
        y_min = int(y_center - bbox_height / 2)
        cv2.rectangle(image, (x_min, y_min), (int(x_min + bbox_width), int(y_min + bbox_height)), (255, 0, 0), 2)
    plt.imshow(image)
    plt.show()

# Define paths to the images and labels directories
images_dir = 'annotated/images'
labels_dir = 'annotated/labels'

# Loop through each image file in the images directory
for image_file in sorted(os.listdir(images_dir)):
    print(image_file)
    image_path = os.path.join(images_dir, image_file)
    label_file = f"{image_file.split('.')[0]}.txt" 
    label_path = os.path.join(labels_dir, label_file)

    if os.path.exists(label_path):
        plot_bounding_boxes(image_path, label_path)
    else:
        print(f"No label found for {image_file}")
    break