import cv2
import pytesseract
import numpy as np
from tkinter import Tk, filedialog


# Set the path to the Tesseract executable if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust if necessary

def select_image():
    """Open a file dialog to select an image file."""
    Tk().withdraw()  # Close the root window
    file_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )
    return file_path

def display_image(title, image):
    """Display an image with a title."""
    plt.figure(figsize=(10, 10))
    plt.title(title)
    if len(image.shape) == 3:
        # Color image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        # Grayscale image
        plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

def process_image(image_path):
    """Process the image and extract text using OCR."""
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to read the image.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Converted to grayscale.")
   

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    print("Applied Gaussian blur to reduce noise.")


    # Apply adaptive thresholding using a Gaussian weighted sum
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    print("Applied adaptive Gaussian thresholding.")
 
    # Perform morphological transformations to improve text visibility
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
    print("Applied morphological transformations.")

    # Perform erosion and dilation to enhance text
    eroded = cv2.erode(morph, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    print("Applied erosion and dilation.")
    

    # Extract text using Tesseract with specific configuration
    custom_config = r'--oem 3 --psm 6'  # OEM 3 means default, PSM 6 assumes a single uniform block of text
    extracted_text = pytesseract.image_to_string(dilated, config=custom_config)
    print("Extracted text using OCR.")
    print(extracted_text)  # Print the extracted text for debugging

    return extracted_text

def main():
    """Main function to handle the receipt processing workflow."""
    # Select an image file
    image_path = select_image()
    if not image_path:
        print("No file selected.")
        return

    # Process the image and extract text
    receipt_details = process_image(image_path)
    # Optional: Save the extracted text to a file or further process it

if __name__ == "__main__":
    main()
