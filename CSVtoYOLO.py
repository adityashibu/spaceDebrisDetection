import pandas as pd
import os
import cv2

# Define paths
base_dir = r"C:\Users\Aditya Uni\.cache\kagglehub\datasets\sadianawar\debris-detection-dataset\versions\1\debris-detection"  # Replace with the path to your dataset
images_dir = os.path.join(base_dir, 'images')
labels_dir = os.path.join(base_dir, 'labels')

# Ensure the labels directory exists
os.makedirs(labels_dir, exist_ok=True)
os.makedirs(os.path.join(labels_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(labels_dir, 'val'), exist_ok=True)

# Paths to CSV files for train and val
train_csv_file = r"C:\Users\Aditya Uni\.cache\kagglehub\datasets\sadianawar\debris-detection-dataset\versions\1\debris-detection\train.csv"  # Replace with your train CSV path
val_csv_file = r"C:\Users\Aditya Uni\.cache\kagglehub\datasets\sadianawar\debris-detection-dataset\versions\1\debris-detection\val.csv"  # Replace with your val CSV path

# Load the CSV files
train_df = pd.read_csv(train_csv_file)
val_df = pd.read_csv(val_csv_file)

# Define class labels for each bounding box (this is just an example)
class_labels = ['debris', 'other']  # Adjust based on your dataset

# Function to convert bounding box to YOLO format
def convert_to_yolo_format(bbox, img_width, img_height):
    # Get the coordinates
    x1, y1, x2, y2 = bbox
    # Ensure that x1 < x2 and y1 < y2
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    
    # Calculate width and height
    width = x2 - x1
    height = y2 - y1
    # Calculate the center of the bounding box
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    # Normalize by image dimensions
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    # Return the YOLO formatted annotation
    return [x_center, y_center, width, height]

# Function to process a CSV file and create label files
def process_csv(df, subfolder):
    # Define paths for the current directory
    image_subdir = os.path.join(images_dir, subfolder)
    label_subdir = os.path.join(labels_dir, subfolder)

    # Ensure the label subdirectory exists
    os.makedirs(label_subdir, exist_ok=True)

    # Process each row in the CSV
    for idx, row in df.iterrows():
        # Get the ImageID and bounding boxes
        image_id = row['ImageID']
        bboxes = eval(row['bboxes'])  # Converting string list to a list of bounding boxes
        
        # Define the path to the image
        img_path = os.path.join(image_subdir, f'{image_id}.jpg')  # Assuming image format is .jpg
        if not os.path.exists(img_path):
            continue  # Skip if the image does not exist

        # Read the image to get its dimensions
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]  # Image dimensions
        
        # Create a list to hold YOLO annotations for this image
        yolo_annotations = []
        
        # Process each bounding box
        for bbox in bboxes:
            yolo_annotations.append('0 ' + ' '.join(map(str, convert_to_yolo_format(bbox, img_width, img_height))))
        
        # Write annotations to a file
        label_file = os.path.join(label_subdir, f'{image_id}.txt')
        with open(label_file, 'w') as f:
            f.write('\n'.join(yolo_annotations))

# Process the train and val CSV files separately
process_csv(train_df, 'train')
process_csv(val_df, 'val')

print('Conversion complete! YOLO annotations saved for train and val sets.')
