import cv2
import pytesseract
import numpy as np
from tkinter import Tk, filedialog
import language_tool_python

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
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return None
    
    image = resize_image(image, 150, cv2.INTER_LINEAR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
    eroded = cv2.erode(morph, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    
    custom_config = r'--oem 3 --psm 6'
    extracted_text = pytesseract.image_to_string(dilated, config=custom_config)
    return extracted_text

def print_text(text):
    """Print the extracted text."""
    print("Extracted Text:")
    print(text)

def format_number(text):
    """Format number with periods as decimal points, ensure two decimal places, and remove any additional spaces."""
    try:
        text = text.replace(',', '.').replace(' ', '')
        number = float(text)
        return f"{number:.2f}"
    except ValueError:
        return text.strip()

def parse_receipt(text):
    """Parse the extracted text into structured data."""
    lines = text.splitlines()
    parsed_data = {"header": [], "items": [], "footer": []}
    
    item_section = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if any(keyword in line for keyword in ["Cashier", "Bill", "TAX INVOICE", "CASH RECEIPT"]):
            parsed_data["header"].append(line)
        elif any(keyword in line for keyword in ["Sub Total", "Total", "Cash", "Change"]):
            formatted_line = ' '.join(format_number(part) if part.replace('.', '', 1).isdigit() else part for part in line.split())
            parsed_data["footer"].append(formatted_line)
        elif any(keyword in line for keyword in ["Qty", "Item", "Amount", "Price", "Description"]):
            item_section = True
            formatted_line = ' '.join(format_number(part) if part.replace('.', '', 1).isdigit() else part for part in line.split())
            parsed_data["items"].append(formatted_line)
        elif item_section and (line.startswith("Total") or line.startswith("Sub Total") or "Cash" in line):
            item_section = False
            formatted_line = ' '.join(format_number(part) if part.replace('.', '', 1).isdigit() else part for part in line.split())
            parsed_data["footer"].append(formatted_line)
        elif item_section:
            formatted_line = ' '.join(format_number(part) if part.replace('.', '', 1).isdigit() else part for part in line.split())
            parsed_data["items"].append(formatted_line)
        else:
            parsed_data["header"].append(line)

    return parsed_data

def check_grammar(text):
    """Check and correct grammar in the text using LanguageTool."""
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    
    if matches:
        print("Grammar Corrections:")
        for match in matches:
            original_text = text[match.offset:match.offset + match.errorLength]
            suggestions = ', '.join(replacement.value for replacement in match.replacements)
            print(f"Original: '{original_text}'")
            print(f"Suggested: {suggestions}\n")
    else:
        print("No grammar mistakes found.")
    
    return corrected_text

def select_image():
    """Open a file dialog to select an image."""
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", ".png;.jpg;.jpeg;.bmp;*.tiff")]
    )
    return file_path

def main():
    """Main function to handle the receipt processing workflow."""
    image_path = select_image()
    if not image_path:
        print("No file selected.")
        return

    extracted_text = process_image(image_path)
    if extracted_text:
        print_text(extracted_text)
        corrected_text = check_grammar(extracted_text)
        print("Corrected Text:")
        print(corrected_text)
        parsed_data = parse_receipt(corrected_text)
        print("Parsed Receipt Data:")
        print(parsed_data)

if _name_ == "_main_":
    main()