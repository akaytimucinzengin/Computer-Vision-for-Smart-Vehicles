# scripts/train_model.py
#
# This script is the heart of the object detection pipeline. It takes the
# prepared and split dataset and uses it to train a YOLOv8 model. The goal
# is to produce a trained model that can accurately detect objects in new images.
#
# The script initializes a YOLOv8 model from a standard configuration (`yolov8n.yaml`).
# It then begins the training process using the `data.yaml` file generated in the
# previous step. This file tells the model where to find the training and
# validation data and what the object classes are.
#
# Key training parameters such as epochs, batch size, and image size are
# defined here but can be easily modified to tune the model's performance.
# After training, the best-performing model weights are saved automatically,
# ready for the evaluation step.

from ultralytics import YOLO
import os

def train_model():
    """
    Trains a YOLOv8 object detection model on a custom dataset.
    
    This function leverages the `ultralytics` library to train a YOLOv8 model.
    It performs the following key actions:
    
    1.  **Model Initialization**: It loads the YOLOv8 model architecture from a
        YAML configuration file (`yolov8n.yaml`), which corresponds to the fast
        and lightweight "nano" version of the model.
        
    2.  **Training Execution**: It calls the `model.train()` method, which
        manages the entire training process. This includes loading the dataset
        as specified in `data/data.yaml`, iterating through the data for a set
        number of epochs, and optimizing the model's weights.
        
    3.  **Hyperparameter Configuration**: The training process is configured
        with key hyperparameters (epochs, batch size, image size). These can be
        tuned to optimize performance for a specific dataset.
        
    4.  **Model Saving**: The `ultralytics` library automatically saves the
        weights of the best-performing model to a file (e.g., `runs/train/exp/weights/best.pt`),
        which can then be used for making predictions.
    """
    # Define the path to the data configuration file.
    yaml_path = 'data/data.yaml'
    if not os.path.exists(yaml_path):
        print(f"Error: {yaml_path} not found.")
        print("Please run the data splitting script first to generate this file.")
        return

    # Load the YOLOv8 model configuration. 'yolov8n.yaml' defines the "nano"
    # model, which is small and fast, making it a good starting point.
    model = YOLO("yolov8n.yaml")

    # Train the YOLOv8 model on the prepared dataset.
    # - `data`: Path to the data.yaml file.
    # - `epochs`: Number of complete passes through the training dataset.
    # - `batch`: Number of samples processed before updating the model weights.
    # - `imgsz`: Image size for training. Images will be resized to this.
    # - `pretrained`: Set to True to start with pre-trained COCO weights,
    #   which is highly recommended for better performance. Set to False to
    #   train from scratch.
    print("Starting model training...")
    model.train(data=yaml_path, epochs=15, batch=8, imgsz=640, pretrained=True)

    print("Model training complete. The best model is saved in the 'runs/train/' directory.")

if __name__ == '__main__':
    train_model()
