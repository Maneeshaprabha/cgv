import cv2
import numpy as np
import pytesseract
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt

# Set the path to the Tesseract executable if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust if necessary

def process_image(image_binary):
    """Process the image to improve OCR accuracy."""
    # Convert binary data to a numpy array
    image_array = np.asarray(bytearray(image_binary), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    if image is None:
        print("Error: Image not found.")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding using a Gaussian weighted sum
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Optional: Perform morphological transformations to improve text visibility
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)

    # Optional: Perform erosion and dilation to enhance text
    eroded = cv2.erode(morph, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)

    # Extract text using Tesseract with specific configuration
    custom_config = r'--oem 3 --psm 6'
    extracted_text = pytesseract.image_to_string(dilated, config=custom_config)

    print("Extracted Text:")
    print(extracted_text)  # Print extracted text to debug

    return extracted_text

def parse_receipt(text):
    """Parse the extracted text into structured data."""
    lines = text.splitlines()
    parsed_data = {"items": []}

    item_section = False

    for line in lines:
        line = line.strip()  # Remove leading/trailing spaces
        if "Item" in line or "Qty" in line or "Description" in line:
            item_section = True
            continue
        
        if item_section:
            # Handle different formats and separators
            parts = line.split()
            if len(parts) >= 2:
                # Check if last part is a price
                last_part = parts[-1].replace(',', '.')
                try:
                    price = float(last_part)
                    item_name = ' '.join(parts[:-1])
                    parsed_data["items"].append((item_name, price))
                except ValueError:
                    continue
    
    return parsed_data

def visualize_sales(parsed_data):
    """Visualize sales data from parsed receipt."""
    items = [item[0] for item in parsed_data["items"]]
    prices = [item[1] for item in parsed_data["items"]]

    if not items or not prices:
        print("No items to display.")
        return

    plt.figure(figsize=(10, 6))
    plt.barh(items, prices, color='skyblue')
    plt.xlabel('Price')
    plt.title('Sales Summary')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def select_image():
    """Open a file dialog to select an image."""
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
    )
    
    # Open the file in binary mode
    if file_path:
        with open(file_path, "rb") as file:
            image_binary = file.read()
        return image_binary
    return None

def main():
    """Main function to run the OCR and visualization."""
    image_binary = select_image()
    if not image_binary:
        print("No file selected.")
        return

    receipt_details = process_image(image_binary)
    if receipt_details:
        parsed_data = parse_receipt(receipt_details)
        if parsed_data["items"]:
            visualize_sales(parsed_data)
        else:
            print("No items to display.")
    else:
        print("No text extracted.")

if __name__ == "__main__":
    main()
