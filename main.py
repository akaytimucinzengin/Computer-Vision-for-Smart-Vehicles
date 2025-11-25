# main.py
#
# This script is the main entry point for the object detection project.
# It orchestrates the entire workflow by calling a series of modular
# scripts, each responsible for a specific part of the pipeline. This
# approach ensures that the code is organized, maintainable, and easy
# to debug.
#
# The project follows a standard machine learning pipeline:
# 1. Data Splitting: Splits the data into training and validation sets.
# 2. Model Training: Trains the YOLOv8 model.
# 3. Model Evaluation: Evaluates the model and makes predictions.
# 4. Results Generation: Creates a CSV file with the prediction results.
#
# To run the entire pipeline, simply execute this script.

from scripts.split_data import split_data
from scripts.train_model import train_model
from scripts.evaluate_model import evaluate_model
from scripts.create_submission import create_submission

def main():
    """
    Main function to run the entire object detection pipeline.
    
    This function orchestrates the complete workflow by calling the individual
    scripts in the correct order. Each script is a self-contained module
    that performs a specific task, making the pipeline easy to understand
    and maintain.
    """
    print("Step 1: Splitting data...")
    split_data()
    print("Data splitting complete.\n")

    print("Step 2: Training model...")
    train_model()
    print("Model training complete.\n")

    print("Step 3: Evaluating model and making predictions...")
    evaluate_model()
    print("Model evaluation complete.\n")

    print("Step 4: Creating submission file...")
    create_submission()
    print("Submission file created.\n")

    print("Pipeline finished successfully!")

if __name__ == '__main__':
    main()
