import cv2
import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Reverse class mapping
REV_CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3: "none"
}

def mapper(val):
    return REV_CLASS_MAP[val]

def predict_image(filepath):
    try:
        # Check if file exists
        if not os.path.exists(filepath):
            print(f"Error: Image file '{filepath}' not found!")
            sys.exit(1)

        # Check if model exists
        model_path = "rock-paper-scissors-model.h5"
        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' not found!")
            sys.exit(1)

        # Load the model
        print("Loading model...")
        model = load_model(model_path)
        
        # Load and preprocess the image
        print("Processing image...")
        img = cv2.imread(filepath)
        if img is None:
            print(f"Error: Could not load image '{filepath}'")
            sys.exit(1)

        # Convert to RGB and resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))  # Resize to match MobileNetV2 input size
        
        # Normalize the image
        img = img.astype('float32') / 255.0
        
        # Make prediction
        print("Making prediction...")
        pred = model.predict(np.array([img]), verbose=0)
        move_code = np.argmax(pred[0])
        move_name = mapper(move_code)
        confidence = pred[0][move_code] * 100

        # Print results
        print("\nResults:")
        print(f"Predicted move: {move_name}")
        print(f"Confidence: {confidence:.2f}%")

        return move_name, confidence

    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)

def main():
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py test_image.jpg")
        sys.exit(1)

    filepath = sys.argv[1]
    predict_image(filepath)

if __name__ == "__main__":
    main()
    