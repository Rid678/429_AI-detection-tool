import torch
import torchvision
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import warnings
import pandas as pd
from math import atan2, cos, sin, sqrt, pi
import sys
import frcnn_utils
import status_shared
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os


class DualOutput:
    def __init__(self, terminal, file):
        self.terminal = terminal
        self.file = file

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()



# Open the log file where you want to redirect the prints
log_file = open("output_log_frcnn.txt", "a")  # 'a' for append, 'w' for overwrite

# Redirect standard output to both terminal and file
sys.stdout = DualOutput(sys.stdout, log_file)


# Suppress specific UserWarnings from torchvision
#comment the below line out in the future to check for warnings, currently it is only used due to asthetics reasons
warnings.filterwarnings("ignore", category=UserWarning, module='torchvision')

# Function to load the model
def load_model(model_path, num_classes):
    # Get the absolute path to the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Combine the script directory with the model file name
    model_path = os.path.join(script_dir, model_path)

    # Check if a model path is provided for a custom-trained model
    if model_path:
        model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
        model.load_state_dict(torch.load(model_path))
    else:
        # Load a model pre-trained on COCO
        model = fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    model.eval()
    return model

# Function to perform inference on a single image
def predict(image, model):
    image_tensor = F.to_tensor(image)
    with torch.no_grad():
        prediction = model([image_tensor])[0]
    return prediction


def find_product_base(image):
    """
    Attempts to find the base of the product by detecting edges and finding the longest horizontal line.
    Returns the angle of this line relative to the horizontal axis.
    """
    # Convert to grayscale and apply edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Use Hough Line Transform to find lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    
    # Placeholder for base line angle calculation
    base_angle = 0
    if lines is not None:
        # Filter out horizontal lines
        horizontal_lines = [line[0] for line in lines if abs(line[0][1] - line[0][3]) < 5]  # Adjust the threshold as needed
        
        if horizontal_lines:
            # Find the longest horizontal line
            longest_line = max(horizontal_lines, key=lambda line: np.linalg.norm(line[:2] - line[2:]))
            x1, y1, x2, y2 = longest_line
            base_angle = np.arctan2(y2 - y1, x2 - x1)  # Angle in radians
    
    return np.degrees(base_angle)  # Convert to degrees


def adjust_label_orientation(label_angle, base_angle):
    """
    Adjusts the label orientation based on the base angle of the product.
    """
    adjusted_angle = label_angle - base_angle
    # Normalize the angle to be between -90 and 90
    if adjusted_angle > 90:
        adjusted_angle -= 180
    elif adjusted_angle < -90:
        adjusted_angle += 180
    return adjusted_angle

# Example usage within the visualization function

def refine_label_detection(image, predicted_boxes):
    refined_boxes = []
    for box in predicted_boxes:
        # Convert box coordinates to integer
        x, y, x_max, y_max = map(int, box)
        
        # Extract the region of interest (ROI) from the image
        roi = image[y:y_max, x:x_max]
        
        # Convert ROI to grayscale and apply adaptive thresholding to find white labels
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        thresholded_roi = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, 11, 2)
        
        # Find contours in the thresholded ROI
        contours, _ = cv2.findContours(thresholded_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:  # Check if any contours were found
            # Find the largest contour, assuming it represents the label
            largest_contour = max(contours, key=cv2.contourArea)
            x_cnt, y_cnt, w, h = cv2.boundingRect(largest_contour)
            
            # Calculate global coordinates for the largest contour by adding offsets
            global_x = x + x_cnt
            global_y = y + y_cnt
            
            # Update refined_boxes with the refined bounding box of the largest contour
            refined_boxes.append((global_x, global_y, global_x + w, global_y + h))
    
    return refined_boxes

# Function to calculate orientation from the bounding box
def get_orientation_from_bbox(box):
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    dy = box[3] - box[1]
    dx = box[2] - box[0]
    angle = np.arctan2(dy, dx)  # Calculate angle in radians
    angle_deg = np.degrees(angle)  # Convert to degrees

    # Normalize angle to [-90, 90] degrees range
    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180

    return (center_x, center_y), angle_deg


# Function to draw arrow axis
def drawAxis(img, p_, q_, color, scale):
    p = list(p_)
    q = list(q_)
 
    angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = sqrt((p[1] - q[1])**2 + (p[0] - q[0])**2)
 
    # Decrease the hypotenuse for smaller arrows
    hypotenuse = hypotenuse * scale  # scale < 1 for smaller
 
    q[0] = p[0] - hypotenuse * cos(angle)
    q[1] = p[1] - hypotenuse * sin(angle)
    # Draw thinner line
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 1, cv2.LINE_AA)  # Thickness decreased to 1
 
    # Adjust scale for smaller arrow heads
    p[0] = q[0] + 5 * cos(angle + pi / 4)  # Decrease size
    p[1] = q[1] + 5 * sin(angle + pi / 4)  # Decrease size
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 1, cv2.LINE_AA)  # Thickness decreased to 1
 
    p[0] = q[0] + 5 * cos(angle - pi / 4)  # Decrease size
    p[1] = q[1] + 5 * sin(angle - pi / 4)  # Decrease size
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 1, cv2.LINE_AA)  # Thickness decreased to 1

def get_orientation_from_contour(pts, img):
    # Calculate the bounding rectangle for the contour
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # Calculate the angle of the bounding box
    angle = rect[2]
    if angle < -45:
        angle += 90

    # Store the angle and the center of the rectangle
    cntr = (int(rect[0][0]), int(rect[0][1]))

    # Optionally, draw the rectangle on the image
    cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
    label = f"Rotation Angle: {angle:.2f} degrees"
    cv2.putText(img, label, (cntr[0] - 50, cntr[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return cntr, angle


# Function to calculate orientation from the bounding box

def getOrientation(pts, img, orientations):
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    angle = rect[2]
    if angle < -45:
        angle = -(90 + angle)  # Correct the angle
    else:
        angle = -angle

    angle = angle % 360
    if angle > 180:
        angle -= 360

    cntr = (int(rect[0][0]), int(rect[0][1]))

    acceptable_ranges = [(88, 90), (178, 180), (-180, -178), (-90, -88)]
    is_acceptable = any(lower <= abs(angle) <= upper for lower, upper in acceptable_ranges)

    if is_acceptable:
        message = "Labels look good to go!"
    else:
        message = "Please check label orientations"

    orientations.append((cntr, angle, message))

    # Optionally, draw the rectangle on the image
    cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

    # Print and draw the angle on the image
    label = f"Rotation Angle: {angle:.2f} degrees"
    cv2.putText(img, label, (cntr[0] + 20, cntr[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    print(label)  # Print rotation angle
    print(message)  # Print message based on angle deviation

    return angle

def process_frcnn_parameters(label_threshold1, label_threshold2, label_threshold3, screw_threshold1, screw_threshold2, screw_threshold3):
    # Process label and screw values received from the UI
    # For example, you can print them or perform any other operations
    print(f"Received label thresholds: {label_threshold1}, {label_threshold2}, {label_threshold3}")
    print(f"Received screw thresholds: {screw_threshold1}, {screw_threshold2}, {screw_threshold3}")

def move_window(event):
    root.geometry(f"+{event.x_root}+{event.y_root}")

def visualize_prediction(image, prediction, label_threshold1, label_threshold2, label_threshold3, screw_threshold1, screw_threshold2, screw_threshold3, threshold=0.5):

    np_image = np.array(image)

    status_value = 0
    
    # Convert the image to grayscale and create an RGB version for colored annotations
    gray_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
    gray_image_rgb = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
    
    plt.figure(figsize=(12, 8))
    
    # Plot the original color image
    plt.subplot(1, 2, 1)
    plt.imshow(np_image)
    plt.title('Original Image (with Bounding Boxes)')
    ax1 = plt.gca()
    ax1.axis('off')

    count_label = 0
    count_screw = 0
    label_counter = 1  # Initialize label counter for naming

    for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
        if score > threshold:
            box = box.detach().cpu().numpy().astype(np.int32)
            x, y, x_max, y_max = box

            if label == 1:  # Labels
                # Increment label counter and counts
                count_label += 1
                
                # Draw red bounding boxes on the original image for labels
                rect = plt.Rectangle((x, y), x_max - x, y_max - y, fill=False, edgecolor='red', linewidth=2)
                ax1.add_patch(rect)

                # Highlight labels on the grayscale image with yellow bounding boxes and text
                cv2.rectangle(gray_image_rgb, (x, y), (x_max, y_max), (0, 255, 255), 2)  # Yellow bounding box
                cv2.putText(gray_image_rgb, f"Label {label_counter}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)  # Yellow label text

                # Find contours and calculate orientation
                roi = gray_image[y:y_max, x:x_max]
                _, binary_roi = cv2.threshold(roi, 220, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    _, orientation_angle = get_orientation_from_contour(largest_contour, gray_image)
                    print(f"Label {label_counter} orientation: {orientation_angle:.2f} degrees")

                label_counter += 1  # Increment only for labels

            elif label == 3:  # Screws
                # Draw green bounding boxes on the original image for screws
                rect = plt.Rectangle((x, y), x_max - x, y_max - y, fill=False, edgecolor='green', linewidth=2)
                ax1.add_patch(rect)
                count_screw += 1

    
    # Check for missing labels and screws using the thresholds for each camera
    camera1_missing_label = max(0, label_threshold1 - count_label)
    camera1_missing_screw = max(0, screw_threshold1 - count_screw)

    camera2_missing_label = max(0, label_threshold2 - count_label)
    camera2_missing_screw = max(0, screw_threshold2 - count_screw)

    camera3_missing_label = max(0, label_threshold3 - count_label)
    camera3_missing_screw = max(0, screw_threshold3 - count_screw)


    if camera1_missing_label > 0:
        print(f"Missing {camera1_missing_label} label(s) for Camera 1")
        status_value = 1
        status_shared.status['value'] = 1  # or 2 for green, 0 for grey
    if camera1_missing_screw > 0:
        print(f"Missing {camera1_missing_screw} screw(s) for Camera 1")
        status_value = 1
        status_shared.status['value'] = 1  # or 2 for green, 0 for grey

    if camera2_missing_label > 0:
        print(f"Missing {camera2_missing_label} label(s) for Camera 2")
        status_value = 1
        status_shared.status['value'] = 1  # or 2 for green, 0 for grey
    if camera2_missing_screw > 0:
        print(f"Missing {camera2_missing_screw} screw(s) for Camera 2")
        status_value = 1
        status_shared.status['value'] = 1  # or 2 for green, 0 for grey

    if camera3_missing_label > 0:
        print(f"Missing {camera3_missing_label} label(s) for Camera 3")
        status_value = 1
        status_shared.status['value'] = 1  # or 2 for green, 0 for grey
    if camera3_missing_screw > 0:
        print(f"Missing {camera3_missing_screw} screw(s) for Camera 3")
        status_value = 1
        status_shared.status['value'] = 1  # or 2 for green, 0 for grey

    if camera1_missing_label == 0 and camera1_missing_screw == 0 and \
       camera2_missing_label == 0 and camera2_missing_screw == 0 and \
       camera3_missing_label == 0 and camera3_missing_screw == 0:
        print("Everything looks good!")
        status_value = 2
        status_shared.status['value'] = 2  # or 2 for green, 0 for grey

    # Plot the grayscale image with colored annotations
    plt.subplot(1, 2, 2)
    plt.imshow(gray_image_rgb)
    plt.title('Grayscale Image with Detected Areas Highlighted')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    print("=" * 40)
    status_value = 0
    #evaluate_image_conditions(status_value)
    status_shared.status['value'] = 0  # or 2 for green, 0 for grey

from multiprocessing import Queue


def get_orientation_from_grayscale(gray_image, contour):
    # Ensure contour is in the correct format and type
    if contour is not None and len(contour) > 0:
        contour = np.array(contour, dtype=np.float32)  # Ensure dtype is float32

        # Calculate the bounding rectangle for the contour
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)  # Convert points to integer for drawing

        # Calculate the angle of the bounding box
        angle = rect[2]
        if angle < -45:
            angle += 90

        # Draw the rectangle on the grayscale image
        cv2.drawContours(gray_image, [box], 0, (0, 255, 0), 2)
        label = f"Rotation Angle: {angle:.2f} degrees"
        cv2.putText(gray_image, label, (int(box[0][0]) - 50, int(box[0][1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return angle
    else:
        # Return a default value or handle the empty contour case
        print("Warning: Empty or invalid contour provided.")
        return None
    
    
# Full file path to save the photo
photo_filename = "captured_photo.jpg"

# Function to take a photo with multiple webcams
def take_photo(label_threshold1, label_threshold2, label_threshold3, screw_threshold1, screw_threshold2, screw_threshold3, total_images):
    print("Press the spacebar to capture and analyze a photo from all three webcams. Press 'Esc' to exit.")

    cap1 = cv2.VideoCapture(2)
    cap2 = cv2.VideoCapture(0)
    cap3 = cv2.VideoCapture(3)

    if not (cap1.isOpened() and cap2.isOpened() and cap3.isOpened()):
        print("Error: Could not open one or more cameras.")
        return None

    while True:
        captured_frames = []

        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            ret3, frame3 = cap3.read()

            if not (ret1 and ret2 and ret3):
                print("Error: Could not capture frames from one or more cameras.")
                return None

            frame1 = cv2.resize(frame1, (320, 240))
            frame2 = cv2.resize(frame2, (320, 240))
            frame3 = cv2.resize(frame3, (320, 240))

            combined_frame = np.vstack((np.hstack((frame1, frame2)), np.hstack((frame3, frame3))))

            cv2.imshow("Live Webcam Feeds", combined_frame)

            key = cv2.waitKey(1)

            if key == 32:  # Spacebar
                print("Photo taken from all three cameras.")

                total_images += 3
                status_shared.total_images = total_images  # Update the shared variable
                
                captured_frames.append((frame1, label_threshold1, screw_threshold1))
                captured_frames.append((frame2, label_threshold2, screw_threshold2))
                captured_frames.append((frame3, label_threshold3, screw_threshold3))

                if len(captured_frames) == 3:
                    break

            if key == 27:  # Escape key
                print("Exiting...")
                cap1.release()
                cap2.release()
                cap3.release()
                cv2.destroyAllWindows()
                return None

        print("Thresholds received in take_photo function:")
        print("For Camera 1 - Label Thresholds:", label_threshold1, "Screw Thresholds:", screw_threshold1)
        print("For Camera 2 - Label Thresholds:", label_threshold2, "Screw Thresholds:", screw_threshold2)
        print("For Camera 3 - Label Thresholds:", label_threshold3, "Screw Thresholds:", screw_threshold3)

        for i, (captured_frame, label_threshold, screw_threshold) in enumerate(captured_frames):
            # Convert the captured frame to PIL Image format
            image = Image.fromarray(cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB))
            # Perform prediction on each image
            prediction = predict(image, model)
            # Visualize the prediction with arrows and check the counts
            visualize_prediction(captured_frame, prediction, label_threshold, label_threshold, label_threshold, screw_threshold, screw_threshold, screw_threshold, threshold=0.5)

        print("Press the spacebar to capture and analyze another set of photos from all three webcams. Press 'Esc' to exit.")




# Load the model - update the path and number of classes
model_path = 'frcnn_model.pth'
num_classes = 4  # Include background as an additional class
model = load_model(model_path, num_classes)

# List of image paths to process
image_paths = [
    r'C:\Users\ridhwanj\Pictures\tI19.jpg',
    r'C:\Users\ridhwanj\Pictures\tI10.jpeg',
    r'C:\Users\ridhwanj\Pictures\captured_photo.jpg',
    r'C:\Users\ridhwanj\Documents\AI project\code\yolo\test3.jpg'
    # Add more image paths as needed
]

# Process each image
#for image_path in image_paths:
    #if image_path != "captured_photo.jpg":
        #image = Image.open(image_path).convert("RGB")
        #prediction = predict(image, model)

        # Visualize the prediction with arrows and check the counts
        #visualize_prediction(image, prediction, threshold=0.5)


# Ask if the user wants to take a photo
#user_response = input("Do you want to take a photo? (yes/no): ")
#if user_response.lower() == "yes":
    #captured_frames = take_photo()
    #if captured_frames is not None:
        # Process each captured frame separately
        #for i, captured_frame in enumerate(captured_frames):
            # Convert the captured frame to PIL Image format
            #image = Image.fromarray(cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB))
            # Perform prediction on each image
            #prediction = predict(image, model)
            # Visualize the prediction with arrows and check the counts
            #visualize_prediction(image, prediction, threshold=0.5)
#else:
    #print("Skipping taking a photo.")


def main():

    # Define default thresholds
    #label_thresholds = [10, 10, 10]  # Example values for three cameras
    #screw_thresholds = [5, 5, 5]     # Example values for three cameras

    # Ask if the user wants to take a photo
    user_response = input("Do you want to take a photo? (yes/no): ")
    if user_response.lower() == "yes":
        captured_frames = take_photo()
        if captured_frames is not None:
            # Process each captured frame separately
            for i, captured_frame in enumerate(captured_frames):
                # Convert the captured frame to PIL Image format
                image = Image.fromarray(cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB))
                # Perform prediction on each image
                prediction = predict(image, model)  # Assuming 'predict' and 'model' are defined in 'new_frcnn_test'
                # Visualize the prediction with arrows and check the counts
                visualize_prediction(image, prediction, label_thresholds, screw_thresholds, threshold=0.5)  # Assuming 'visualize_prediction' is defined in 'new_frcnn_test'
    else:
        print("Skipping taking a photo.")

    # Ensure the log file is properly closed
    log_file.close()

if __name__ == "__main__":
    main()

