import cv2
import numpy as np
import pytesseract
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt

# Set the path to the Tesseract executable if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def resize_image(image, scale_percent, interpolation_method):
    """Resize the image with specified interpolation method."""
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=interpolation_method)
    return resized_image

def process_image(image_path):
    """Process the image to improve OCR accuracy."""
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Image not found.")
        return None

    # Define interpolation methods
    interpolation_methods = {
        'Nearest': cv2.INTER_NEAREST,
        'Linear': cv2.INTER_LINEAR,
        'Cubic': cv2.INTER_CUBIC,
        'Lanczos4': cv2.INTER_LANCZOS4
    }

    scale_percent = 200  # Scale image to 200% of original size
    
    for name, method in interpolation_methods.items():
        resized_image = resize_image(image, scale_percent, method)
        print(f"Image resized using {name} interpolation.")
        break  # Remove this line if you want to use all interpolation methods

    # Convert to grayscale
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
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
            parts = line.split()
            if len(parts) >= 2:
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

    # Define colors for specific terms
    color_map = {
        'Subtotal': 'lightcoral',
        'Total': 'lightcoral',
        'Cash': 'Yellow',
        'Change': 'lightgreen'
    }

    colors = ['skyblue'] * len(items)
    for i, item in enumerate(items):
        for keyword, color in color_map.items():
            if keyword in item:
                colors[i] = color
                break

    plt.figure(figsize=(10, 6))
    plt.barh(items, prices, color=colors)
    plt.xlabel('Price')
    plt.title('Sales Summary')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Print text-based bar chart
    max_length = max(len(item) for item in items)
    max_price = max(prices)
    scale_factor = 50 / max_price  # Scale prices to fit within a reasonable range

    print("\nSales Summary:")
    for item, price, color in zip(items, prices, colors):
        bar = "#" * int(price * scale_factor)
        print(f"{item.ljust(max_length)} | {bar} ({price:.2f})")

    
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
    """Main function to run the OCR and visualization."""
    image_path = select_image()
    if not image_path:
        print("No file selected.")
        return

    receipt_details = process_image(image_path)
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

