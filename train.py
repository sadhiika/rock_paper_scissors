import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import os
import sys

# Configure GPU memory growth
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

# Constants
IMG_SAVE_PATH = 'image_data'
CLASS_MAP = {
    "rock": 0,
    "paper": 1,
    "scissors": 2,
    "none": 3
}
NUM_CLASSES = len(CLASS_MAP)
IMG_SIZE = 224  # MobileNetV2 default input size

def mapper(val):
    return CLASS_MAP[val]

def get_model():
    try:
        # Use MobileNetV2 as base model (smaller and faster than SqueezeNet)
        base_model = MobileNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(NUM_CLASSES, activation='softmax')(x)
        
        # Create final model
        model = Model(inputs=base_model.input, outputs=predictions)
        return model
    except Exception as e:
        print(f"Error creating model: {e}")
        sys.exit(1)

def load_data():
    dataset = []
    try:
        for directory in os.listdir(IMG_SAVE_PATH):
            path = os.path.join(IMG_SAVE_PATH, directory)
            if not os.path.isdir(path):
                continue
                
            print(f"Loading images from {directory}...")
            for item in os.listdir(path):
                if item.startswith("."):
                    continue
                    
                img_path = os.path.join(path, item)
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"Warning: Could not load image {img_path}")
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                dataset.append([img, directory])
                
        return dataset
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def main():
    # Check if image directory exists
    if not os.path.exists(IMG_SAVE_PATH):
        print(f"Error: Directory {IMG_SAVE_PATH} not found!")
        sys.exit(1)
        
    print("Loading dataset...")
    dataset = load_data()
    
    if not dataset:
        print("Error: No images found in the dataset!")
        sys.exit(1)
    
    print(f"Total images loaded: {len(dataset)}")
    
    # Split data and labels
    data, labels = zip(*dataset)
    labels = list(map(mapper, labels))
    
    # Convert to numpy arrays and normalize
    X = np.array(data, dtype='float32') / 255.0
    y = to_categorical(labels, NUM_CLASSES)
    
    print("\nData shape:", X.shape)
    print("Labels shape:", y.shape)
    
    # Create and compile model
    print("\nCreating model...")
    model = get_model()
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the model
    print("\nStarting training...")
    try:
        history = model.fit(
            X, y,
            epochs=10,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        # Save the model
        model_path = "rock-paper-scissors-model.h5"
        model.save(model_path)
        print(f"\nModel saved to {model_path}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
    