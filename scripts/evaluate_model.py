# scripts/evaluate_model.py
#
# This script is the third step in the pipeline. It evaluates the performance
# of the trained model and uses it to make predictions on the test dataset.
# This provides a quantitative measure of the model's accuracy and its ability
# to generalize to new, unseen data.
#
# The script performs two main tasks:
# 1.  **Prediction on Test Set**: It loads the best-performing model weights
#     saved during training and runs inference on all images in `data/test/images`.
#     The prediction results, including bounding boxes and class labels, are
#     saved to a new directory under `runs/detect/`.
#
# 2.  **Performance Validation**: The script also runs the `model.val()` function,
#     which calculates key object detection metrics like mean Average Precision (mAP)
#     on the validation set. This provides a standardized way to assess the
#     model's performance.

from ultralytics import YOLO
import os

def get_latest_train_run():
    """Finds the latest training experiment directory and returns the path to the best model weights."""
    runs_dir = 'runs/train'
    if not os.path.exists(runs_dir):
        return None
    
    # List all experiment directories (e.g., 'exp', 'exp2', ...)
    exp_dirs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) if d.startswith('exp')]
    if not exp_dirs:
        return None
        
    # Find the most recently modified experiment directory
    latest_exp_dir = max(exp_dirs, key=os.path.getmtime)
    weights_path = os.path.join(latest_exp_dir, 'weights', 'best.pt')
    
    if os.path.exists(weights_path):
        return weights_path
    else:
        print(f"Warning: 'best.pt' not found in the latest training run: {latest_exp_dir}")
        return None

def evaluate_model():
    """
    Evaluates the trained YOLOv8 model and makes predictions on the test set.
    
    This function handles the post-training phase by assessing the model's
    performance and generating predictions on unseen test images.
    
    The key operations are:
    1.  **Model Loading**: It dynamically finds and loads the best-performing
        model from the most recent training run.
        
    2.  **Prediction**: It runs the model on the test dataset located in
        `data/test/images`. The `model.predict()` method outputs the
        detected objects, including their class, bounding box coordinates,
        and confidence score. Predictions are saved for the next step.
        
    3.  **Performance Validation**: It calculates the mean Average Precision (mAP)
        on the validation set to provide a quantitative measure of the model's
        accuracy.
    """
    # Find the path to the best model from the latest training run.
    model_path = get_latest_train_run()
    if model_path is None:
        print("Error: No trained model found. Please run the training script first.")
        return

    print(f"Loading model from: {model_path}")
    # Load the trained YOLOv8 model.
    model = YOLO(model_path)

    # Define the directory for the test images.
    test_images_dir = "data/test/images"
    if not os.path.exists(test_images_dir):
        print(f"Error: Test images directory not found at '{test_images_dir}'.")
        print("Please place your test images in this directory.")
        return

    # Make predictions on the test images.
    # - `source`: Directory containing the test images.
    # - `iou`: IoU threshold for Non-Maximum Suppression (NMS).
    # - `conf`: Confidence threshold for filtering detections.
    # - `save_txt`: Save the prediction bounding boxes to .txt files.
    print(f"Running predictions on images in '{test_images_dir}'...")
    results = model.predict(source=test_images_dir, iou=0.5, conf=0.75, save=True, save_txt=True)
    print("Predictions complete. Results saved in the 'runs/detect/' directory.")

    # Evaluate the model's performance on the validation set.
    yaml_path = 'data/data.yaml'
    print("Evaluating model performance on the validation set...")
    val_results = model.val(data=yaml_path, iou=0.5)
    print("mAP@0.5 results:", val_results.box.map50)

if __name__ == '__main__':
    evaluate_model()
