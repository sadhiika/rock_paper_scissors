desc = '''Script to gather data images with a particular label.

Usage: python gather_images.py <label_name> <num_samples>

The script will collect <num_samples> number of images and store them
in its own directory.

Only the portion of the image within the box displayed
will be captured and stored.

Press 'a' to start/pause the image collecting process.
Press 'q' to quit.

'''
import cv2
import os
import sys

# Help description
desc = """
Usage:
python gather_images.py <label_name> <num_samples>

Example:
python gather_images.py rock 100
"""

# Check command line arguments
try:
    label_name = sys.argv[1]
    num_samples = int(sys.argv[2])
except IndexError:
    print("Error: Missing arguments.")
    print(desc)
    sys.exit(1)
except ValueError:
    print("Error: Second argument must be a number.")
    print(desc)
    sys.exit(1)

# Setup paths - using Windows path style
IMG_SAVE_PATH = 'image_data'
IMG_CLASS_PATH = os.path.join(IMG_SAVE_PATH, label_name)

# Create directories
for directory in [IMG_SAVE_PATH, IMG_CLASS_PATH]:
    os.makedirs(directory, exist_ok=True)

# Initialize camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Added cv2.CAP_DSHOW for Windows

if not cap.isOpened():
    print("Error: Could not open camera.")
    sys.exit(1)

# Set camera properties for better performance on Windows
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

start = False
count = 0

print("\nInstructions:")
print("Press 'a' to start/pause image collection")
print("Press 'q' to quit")
print(f"Collecting {num_samples} images for '{label_name}'\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        continue

    if count == num_samples:
        break

    # Draw rectangle for ROI (Region of Interest)
    cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)

    if start:
        try:
            roi = frame[100:500, 100:500]
            save_path = os.path.join(IMG_CLASS_PATH, f'{count + 1}.jpg')
            cv2.imwrite(save_path, roi)
            count += 1
        except Exception as e:
            print(f"Error saving image: {e}")
            continue

    # Add text to frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Collecting {count}/{num_samples}",
                (5, 50), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    
    # Show recording status
    status = "Recording" if start else "Paused"
    cv2.putText(frame, status,
                (5, 80), font, 0.7, (0, 255, 0) if start else (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Collecting images", frame)

    # Handle key presses
    key = cv2.waitKey(10) & 0xFF
    if key == ord('a'):
        start = not start
        status_msg = "Started" if start else "Paused"
        print(f"{status_msg} recording...")
    elif key == ord('q'):
        print("\nRecording stopped by user.")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

print(f"\nCollection completed:")
print(f"- {count} image(s) saved to {IMG_CLASS_PATH}")