import easyocr
import cv2
import numpy as np
import pandas as pd
from math import atan2, cos, sin, sqrt, pi
import re  # Import the regular expression module

# Load a pre-trained model for a specific language (e.g., 'en' for English)
reader = easyocr.Reader(['en'])

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

def getOrientation(pts, img, orientations):
    # Calculate the bounding rectangle for the contour
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # Calculate the angle of the bounding box
    angle = rect[2]

    if angle < -45:
        angle = -(90 + angle)  # Correct the angle
    else:
        angle = -angle

    # Normalize the angle to be between -180 to 180
    angle = angle % 360
    if angle > 180:
        angle -= 360

    # Store the angle and the center of the rectangle
    cntr = (int(rect[0][0]), int(rect[0][1]))

    # Determine message based on angle deviation
    # Accept angles within a range around 0, 90, -90, 180, and -180 degrees
    acceptable_ranges = [(88, 90), (178, 180), (-180, -178), (-90, -88)]
    is_acceptable = any(lower <= abs(angle) <= upper for lower, upper in acceptable_ranges)

    if is_acceptable:
        message = "Labels look good to go!"
    else:
        message = "Please check label orientations"

    orientations.append((cntr, angle, message))  # Append a tuple of three elements

    # Optionally, draw the rectangle on the image
    cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

    # Print and draw the angle on the image
    label = f"Rotation Angle: {angle:.2f} degrees"
    cv2.putText(img, label, (cntr[0] + 20, cntr[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    print(label)  # Print rotation angle
    print(message)  # Print message based on angle deviation

    return angle




# Load the image
image = cv2.imread('test3.jpg')

# Resize the image for better OCR results
max_height = 800
scale_factor = max_height / image.shape[0]
new_width = int(image.shape[1] * scale_factor)
new_height = int(image.shape[0] * scale_factor)
resized_image = cv2.resize(image, (new_width, new_height))

# Perform OCR on the resized image
results = reader.readtext(resized_image)

# Load the Excel file with the OCR combinations as strings
df = pd.read_excel('OCR_EXD.xlsx', dtype=str)

# Initialize a list to store matched lines
matched_lines = []

# Initialize a list to store all lines
all_lines = []

# Initialize a list to store found reference numbers
found_refs = []

# Iterate through the results and process the text
for result in results:
    text = result[1]
    
    # Append the text to the list of all lines
    all_lines.append(text)

    # Check if the line exists in the Excel file
    if text in df.values:
        matched_lines.append(text)

    # Search for the reference number pattern with more flexibility
    ref_pattern = r"(Ref:No:|RER:No:)(1S|15)[0-9]+"  # Adjusted to capture "RER:No:" as well
    ref_matches = re.findall(ref_pattern, text, re.IGNORECASE)  # Use re.IGNORECASE to make the search case-insensitive
    if ref_matches:
        for match in ref_matches:
            # Extract and print the entire matched string
            matched_string = text[text.find(match[0]):]  # Extract the string starting from the match
            # Correct "RER:No:15" or "Ref:No:15" to "Ref:No:1S"
            corrected_string = re.sub(r'^(RER:No:|Ref:No:)15', 'Ref:No:1S', matched_string, flags=re.IGNORECASE)
            found_refs.append(corrected_string)  # Append the corrected substring

# Print the matched lines to the terminal
for line in matched_lines:
    print(line)

# Print found reference numbers
for ref in found_refs:
    print("Found reference number:", ref)

# Write all lines to a text file
with open('OCR_output.txt', 'w') as file:
    file.write('\n'.join(all_lines))

# Convert the resized image to grayscale
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Threshold the grayscale image to obtain a binary image of white labels
_, binary_image = cv2.threshold(gray_image, 220, 255, cv2.THRESH_BINARY)

# List to store the orientation of labels
label_orientations = []

# Find contours on the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours and analyze orientation for white labels
for c in contours:
    area = cv2.contourArea(c)
    if area < 3700 or 100000 < area:
        continue
    cv2.drawContours(resized_image, [c], -1, (0, 0, 255), 2)
    getOrientation(c, resized_image, label_orientations)

# Check if any orientations were stored
if label_orientations:
    # Sort the label orientations by y-coordinate
    sorted_orientations = sorted(label_orientations, key=lambda x: x[0][1], reverse=True)
    
    # Get the last orientation details
    last_orientation = sorted_orientations[0]
    cntr, rotation_angle_deg, message = last_orientation

    # Print the last orientation angle
    print(f"Last Label Orientation: {rotation_angle_deg} degrees")

    # Print the last orientation message
    print(f"Last Label Message: {message}")

    # Optionally, draw the last orientation angle on the image
    label_position = (cntr[0], resized_image.shape[0] - 10)  # Position at the bottom
    cv2.putText(resized_image, f"Last Label Orientation: {rotation_angle_deg} degrees", label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

# Display the image with detected white labels
cv2.imshow('White Label Detection', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
