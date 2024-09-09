import cv2
import pytesseract
import numpy as np
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt


# Set the path to the Tesseract executable if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust if necessary

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

    # Display original image
    display_image("Original Image", image)

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
        display_image(f"Resized Image ({name})", resized_image)

    # Convert to grayscale
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    print("Converted to grayscale.")
    display_image("Grayscale Image", gray)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    print("Applied Gaussian blur to reduce noise.")
    display_image("Blurred Image", blurred)

    # Apply adaptive thresholding using a Gaussian weighted sum
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    print("Applied adaptive Gaussian thresholding.")
    display_image("Adaptive Thresholding", adaptive_thresh)

    # Perform morphological transformations to improve text visibility
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
    print("Applied morphological transformations.")
    display_image("Morphological Transformations", morph)

    # Perform erosion and dilation to enhance text
    eroded = cv2.erode(morph, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    print("Applied erosion and dilation.")
    display_image("Eroded Image", eroded)
    display_image("Dilated Image", dilated)

    # Extract text using Tesseract with specific configuration
    custom_config = r'--oem 3 --psm 6'  # OEM 3 means default, PSM 6 assumes a single uniform block of text
    extracted_text = pytesseract.image_to_string(dilated, config=custom_config)
    print("Extracted text using OCR.")

    return extracted_text

def format_number(text):
    """Format number with periods as decimal points, ensure two decimal places, and remove any additional spaces."""
    try:
        # Replace commas with periods and remove spaces
        text = text.replace(',', '.').replace(' ', '')
        # Convert text to a float
        number = float(text)
        # Format with two decimal places
        formatted_number = f"{number:.2f}"
        return formatted_number
    except ValueError:
        # If conversion fails, return the original text
        return text.strip()

def parse_receipt(text):
    """Parse the extracted text into structured data."""
    lines = text.splitlines()
    parsed_data = {"header": [], "items": [], "footer": []}
    
    item_section = False

    for line in lines:
        if "Cashier" in line or "Bill" in line:
            parsed_data["header"].append(line.strip())
        elif "Sub Total" in line or "Cash" in line or "Change" in line:
            # Split footer items to detect and format numbers
            parts = line.split()
            formatted_parts = [format_number(part) if part.replace('.', '', 1).isdigit() else part for part in parts]
            parsed_data["footer"].append(" ".join(formatted_parts))
        elif line.strip() and not item_section:
            parsed_data["header"].append(line.strip())
        elif line.startswith("#"):
            item_section = True
            parsed_data["items"].append(line.strip())
        elif item_section:
            # Split items to detect and format numbers
            parts = line.split()
            formatted_parts = []
            for part in parts:
                if part.replace('.', ',', 1).isdigit():
                    # Check if it is a quantity (integer without decimals) or a price (float with two decimals)
                    if '.' in part:
                        formatted_parts.append(format_number(part))  # Price should have two decimal places
                    else:
                        formatted_parts.append(f"{int(part):.0f}")  # Quantity should not have decimals
                else:
                    formatted_parts.append(part)
            parsed_data["items"].append(" ".join(formatted_parts))
    
    return parsed_data

def print_receipt_table(parsed_data):
    """Print the parsed data in a table format to resemble a receipt."""
    print("\n" + "=" * 35)
    for header in parsed_data["header"]:
        print(f"{header:^35}")
    
    print("\n" + "-" * 35)
    for item in parsed_data["items"]:
        print(f"{item}")
    
    print("-" * 35)
    for footer in parsed_data["footer"]:
        print(f"{footer:<35}")
    print("=" * 35 + "\n")

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
    receipt_details = process_image(image_path)

    # Display the extracted text in a table format
    if receipt_details:
        parsed_data = parse_receipt(receipt_details)
        print_receipt_table(parsed_data)

if __name__ == "__main__":
    main()
