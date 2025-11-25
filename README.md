# High-Performance YOLOv8 Training Pipeline for Custom Object Detection

This project provides a powerful, reusable, and easy-to-use pipeline for training state-of-the-art YOLOv8 object detection models on your own custom datasets. The code is professionally structured and heavily commented to serve as a robust template for any custom object detection task, enabling you to achieve high-accuracy results with minimal setup.

Whether you are a researcher, a student, or a developer, this pipeline provides a solid foundation for building and deploying advanced computer vision systems.

## Key Features

-   **Modular & Professional Structure**: The entire pipeline is broken down into clean, single-responsibility Python scripts, making it easy to understand, modify, and maintain.
-   **Fully Automated**: Run the entire workflow—from data splitting to model training and results generation—with a single command.
-   **Easily Customizable**: Designed to be easily adapted for any custom dataset. Simply structure your data as described, update the `data.yaml` configuration, and start training.
-   **High Performance**: Leverages the state-of-the-art YOLOv8 model to achieve high accuracy and fast inference speeds.
-   **Extensive Documentation**: Includes detailed comments and a comprehensive README to guide you through every step of the process.

## Project Structure

The project is organized to be as intuitive as possible:

```
/
|-- main.py                 # Main orchestrator to run the entire pipeline
|-- requirements.txt        # All Python dependencies
|-- README.md               # This guide
|-- scripts/
|   |-- split_data.py       # Automatically splits your data into training and validation sets
|   |-- train_model.py      # Trains the YOLOv8 model on your custom data
|   |-- evaluate_model.py   # Evaluates the model and runs predictions on test images
|   |-- create_submission.py# Generates a clean CSV file with all prediction results
|-- data/
|   |-- train/
|   |   |-- images/         # Place your training images here
|   |   |-- labels/         # Place your training YOLO .txt labels here
|   |-- test/
|   |   |-- images/         # Place your test images here
|   |-- (val/ will be created automatically)
```

## Getting Started

Follow these steps to train a model on your own dataset.

### 1. Prerequisites

-   Python 3.8 or higher
-   `pip` for package management

### 2. Installation

First, clone this repository and navigate into the project directory:
```bash
git clone https://github.com/akaytimucinzengin/Computer-Vision-for-Smart-Vehicles.git
cd Computer-Vision-for-Smart-Vehicles
```

Next, it's highly recommended to create a virtual environment to manage dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

Finally, install all the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Data Preparation and Configuration

This pipeline is designed to work with any custom dataset. To prepare your data, follow these two simple steps:

**A. Structure Your Data Folders**

Organize your image and label files into the following structure inside the `data/` directory:

1.  `data/train/images/`: Place all your initial training images (`.jpg`, `.png`, etc.) here.
2.  `data/train/labels/`: Place the corresponding YOLO-formatted annotation files (`.txt`) here. Each image must have a matching `.txt` file.
3.  `data/test/images/`: Place the images you want to run predictions on after the model is trained.

*What is YOLO format?* For each image, you need a `.txt` file with the same name. Each line in the file represents one object with `<class_id> <x_center> <y_center> <width> <height>`, where coordinates are normalized (between 0 and 1).

**B. Configure Your `data.yaml`**

The `data.yaml` file is the most important configuration file for your model. The `split_data.py` script will generate it for you, but **you must verify its contents**. After running the data splitting step, open `data/data.yaml` and make sure the following fields match your dataset:

-   `nc`: The total number of object classes.
-   `names`: A list of the names for each class, in order. The first name corresponds to class ID `0`, the second to class ID `1`, and so on.

**Example `data.yaml` for a custom dataset:**
```yaml
train: /path/to/your/project/data/train/images
val: /path/to/your/project/data/val/images

# Number of classes
nc: 2

# Class names
names: ['person', 'bicycle']
```

## How to Run the Pipeline

With your data in place and configured, you can run the entire pipeline with a single command:

```bash
python main.py
```

This will automatically:
1.  **Split your data** into training and validation sets (creating the `data/val/` directory).
2.  **Train the YOLOv8 model** on your dataset.
3.  **Evaluate the trained model**'s performance.
4.  **Run predictions** on all images in your `data/test/images` folder.
5.  **Generate a `results.csv` file** with all the detections from your test images.

## Key Technologies

-   **Python**: The core language of the project.
-   **PyTorch & ultralytics**: The powerful deep learning framework and library behind YOLOv8.
-   **Pandas**: Used for creating the final results CSV file.
-   **Pillow (PIL)**: Used for image processing tasks.

This well-structured and documented pipeline empowers you to train high-performance object detection models on any custom dataset with confidence.
