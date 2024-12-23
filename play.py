import cv2
import numpy as np
from random import choice
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Constants
WINDOW_NAME = "Rock Paper Scissors"
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
IMG_SIZE = 224  # Match with training model size

# Class mapping
REV_CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3: "none"
}

def mapper(val):
    return REV_CLASS_MAP[val]


def calculate_winner(move1, move2):
    if move1 == move2:
        return "Tie"

    if move1 == "rock":
        if move2 == "scissors":
            return "User"
        if move2 == "paper":
            return "Computer"

    if move1 == "paper":
        if move2 == "rock":
            return "User"
        if move2 == "scissors":
            return "Computer"

    if move1 == "scissors":
        if move2 == "paper":
            return "User"
        if move2 == "rock":
            return "Computer"


def load_game_model():
    try:
        model_path = "rock-paper-scissors-model.h5"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found!")
        return load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def initialize_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend
    if not cap.isOpened():
        raise Exception("Could not open camera!")
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    return cap

def load_computer_move_images():
    images = {}
    for move in ['rock', 'paper', 'scissors']:
        try:
            path = f"images/{move}.png"
            if not os.path.exists(path):
                print(f"Warning: Icon file for {move} not found at {path}")
                continue
                
            img = cv2.imread(path)
            if img is None:
                print(f"Warning: Could not load icon for {move}")
                continue
                
            # Resize image to match the computer's box size
            images[move] = cv2.resize(img, (300, 300))
            print(f"Successfully loaded {move} icon")
            
        except Exception as e:
            print(f"Error loading {move} icon: {e}")
            
    if not images:
        print("No computer move icons could be loaded!")
    
    return images

def main():
    try:
        # Initialize model and camera
        model = load_game_model()
        if model is None:
            return
        
        cap = initialize_camera()
        computer_move_images = load_computer_move_images()
        
        prev_move = None
        winner = "Waiting..."
        computer_move_name = "none"
        
        print("\nInstructions:")
        print("1. Show your hand gesture in the left rectangle")
        print("2. Press 'q' to quit")
        print("3. Make sure you have good lighting\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                continue
                
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Draw rectangles
            # User rectangle (left side)
            cv2.rectangle(frame, (50, 100), (350, 400), (255, 255, 255), 2)
            # Computer rectangle (right side)
            cv2.rectangle(frame, (750, 100), (1050, 400), (255, 255, 255), 2)
            
            # Extract and process user's move
            roi = frame[100:400, 50:350]
            if roi.size == 0:
                continue
                
            img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            
            # Predict move
            pred = model.predict(np.array([img]), verbose=0)
            move_code = np.argmax(pred[0])
            user_move_name = mapper(move_code)
            
            # Update computer's move only when user's move changes
            if prev_move != user_move_name:
                if user_move_name != "none":
                    computer_move_name = choice(['rock', 'paper', 'scissors'])
                    winner = calculate_winner(user_move_name, computer_move_name)
                else:
                    computer_move_name = "none"
                    winner = "Waiting..."
            
            prev_move = user_move_name
            
            # Display information
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Display moves
            cv2.putText(frame, f"Your Move: {user_move_name}",
                       (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Computer's Move: {computer_move_name}",
                       (750, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display winner
            winner_color = (0, 255, 0) if winner == "User" else (0, 0, 255) if winner == "Computer" else (255, 255, 255)
            cv2.putText(frame, f"Winner: {winner}",
                       (450, 50), font, 1, winner_color, 2, cv2.LINE_AA)
            
            # Display computer's move image
            if computer_move_name != "none" and computer_move_name in computer_move_images:
                icon = computer_move_images[computer_move_name]
                if icon is not None:
                    frame[100:400, 750:1050] = icon
            
            # Show frame
            cv2.imshow(WINDOW_NAME, frame)
            
            # Check for quit
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

    