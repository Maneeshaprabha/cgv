
import cv2
import pytesseract
import numpy as np
from tkinter import Tk, filedialog


# Set the path to the Tesseract executable if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust if necessary

def resize_image(image, scale_percent, interpolation_method):
    """Resize the image with specified interpolation method."""
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    resized_image = cv2.resize(image, dim, interpolation=interpolation_method)
    return resized_image

def process_image(image_path):
    """Process the image to improve OCR accuracy."""
    # Read the image from the file path
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Image not found.")
        return None
    
    # Optionally resize the image (e.g., 150% of the original size, using linear interpolation)
    image = resize_image(image, 150, cv2.INTER_LINEAR)

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

def print_text(text):
    """Print the extracted text."""
    print("Extracted Text:")
    print(text)

def parse_receipt(text):
    """Parse the extracted text into structured data."""
    lines = text.splitlines()
    parsed_data = {"header": [], "items": [], "footer": []}
    
    item_section = False
    in_item = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Identify sections based on keywords and patterns
        if "Cashier" in line or "Bill" in line or "TAX INVOICE" in line or "CASH RECEIPT" in line:
            parsed_data["header"].append(line)
        elif "Sub Total" in line or "Total" in line or "Cash" in line or "Change" in line:
            parsed_data["footer"].append(line)
        elif any(keyword in line for keyword in ["Qty", "Item", "Amount", "Price", "Description"]):
            item_section = True
            parsed_data["items"].append(line)
            in_item = True
        elif item_section and (line.startswith("Total") or line.startswith("Sub Total") or "Cash" in line):
            item_section = False
            parsed_data["footer"].append(line)
        elif item_section:
            parsed_data["items"].append(line)
        else:
            parsed_data["header"].append(line)

    return parsed_data

def select_image():
    """Open a file dialog to select an image."""
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
    )
    
    return file_path

def main():
    """Main function to handle the receipt processing workflow."""
    # Select an image file
    image_path = select_image()
    if not image_path:
        print("No file selected.")
        return

    # Process the image and extract text
    extracted_text = process_image(image_path)

    # Display the extracted text
    if extracted_text:
        print_text(extracted_text)
        
        # Parse the extracted text
        parsed_data = parse_receipt(extracted_text)
        print("Parsed Receipt Data:")
        print(parsed_data)

if __name__ == "__main__":
    main()
