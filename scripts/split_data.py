# scripts/split_data.py
#
# This script is the first step in the object detection pipeline. It splits the
# user's dataset into training and validation sets. This is a fundamental
# practice in machine learning, allowing the model to be evaluated on data it
# has not seen during training, which helps prevent overfitting.
#
# The script assumes that all initial labeled data (images and YOLO-formatted
# .txt files) are located in `data/train/images` and `data/train/labels`.
# It then randomly samples a portion of this data (defaulting to 20%) and moves
# it into `data/val/images` and `data/val/labels`.
#
# After splitting the data, this script creates the `data.yaml` file. This
# is a configuration file required by YOLOv8 that specifies the paths to the
# training and validation data, the number of classes, and their names. This
# file is essential for the training process.

import os
import shutil
import random

def split_data():
    """
    Splits the dataset into training and validation sets.
    
    This function partitions the user's dataset into two subsets: one for
    training the model and one for validating its performance. This separation
    is essential for assessing the model's ability to generalize to new,
    unseen data.
    
    The process includes the following steps:
    1.  **Directory Definition**: The function defines the expected directory
        structure for a custom dataset: `data/train` for initial data and
        `data/val` for the validation split.
        
    2.  **Directory Creation**: It ensures that the validation directories
        (`data/val/images` and `data/val/labels`) exist, creating them if
        they don't.
        
    3.  **Random Sampling**: It randomly selects a percentage of the images
        from the training directory to be moved to the validation directory.
        
    4.  **File Moving**: The selected images and their corresponding YOLO
        annotation files are moved to the validation directories, physically
        separating the two sets.
        
    5.  **YAML Configuration File Creation**: It generates the `data.yaml`
        file required by YOLO, which contains the paths to the data and
        metadata about the object classes.
    """
    print("Assuming all initial data is in 'data/train'...")

    # Define the simplified directory structure for the dataset.
    train_images_dir = 'data/train/images'
    train_labels_dir = 'data/train/labels'
    val_images_dir = 'data/val/images'
    val_labels_dir = 'data/val/labels'

    # Ensure the source directories exist.
    if not os.path.exists(train_images_dir) or not os.path.exists(train_labels_dir):
        print(f"Error: Source directories not found.")
        print("Please ensure your images are in 'data/train/images' and your labels are in 'data/train/labels'.")
        return

    # Create the validation directories if they do not already exist.
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # Get a list of all images in the training directory.
    all_images = [f for f in os.listdir(train_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Calculate 20% of the images for the validation set.
    val_size = int(len(all_images) * 0.2)
    if val_size == 0 and len(all_images) > 0:
        val_size = 1 # Ensure at least one image is used for validation if the dataset is very small.

    val_images = random.sample(all_images, val_size)

    # Move the selected images and their corresponding labels to the validation directories.
    for img_name in val_images:
        img_path = os.path.join(train_images_dir, img_name)
        
        # Determine the label name from the image name.
        base_name = os.path.splitext(img_name)[0]
        label_name = f"{base_name}.txt"
        label_path = os.path.join(train_labels_dir, label_name)

        # Move the image and label file to the validation set.
        shutil.move(img_path, os.path.join(val_images_dir, img_name))
        if os.path.exists(label_path):
            shutil.move(label_path, os.path.join(val_labels_dir, label_name))

    print(f"{len(val_images)} image(s) and their labels moved to the validation set.")

    # Create the data.yaml file required for YOLO training.
    # The paths are converted to absolute paths to ensure YOLO can find them.
    # IMPORTANT: Update nc and names to match YOUR custom dataset.
    yaml_content = f"""
train: {os.path.abspath(train_images_dir)}
val: {os.path.abspath(val_images_dir)}

# Number of classes
nc: 3

# Class names
names: ['car', 'traffic sign', 'pedestrian']
"""

    # Save the content to the data.yaml file.
    yaml_path = 'data/data.yaml'
    with open(yaml_path, 'w') as yaml_file:
        yaml_file.write(yaml_content.strip())

    print(f"data.yaml file created at: {yaml_path}")
    print("IMPORTANT: Please verify the 'nc' and 'names' fields in data.yaml match your dataset.")

if __name__ == '__main__':
    split_data()
