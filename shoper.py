import cv2
import pytesseract
from tkinter import Tk, filedialog

# Set the path to the Tesseract executable if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust if necessary

def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Image not found.")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Converted to grayscale.")

    # Apply thresholding for better OCR results
    _, binary_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    print("Applied binarization.")

    # Extract text from the binarized image
    extracted_text = pytesseract.image_to_string(binary_image)
    print("Extracted text using OCR.")

    return extracted_text

def select_image():
    # Open a file dialog to select an image
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
    )
    return file_path

def main():
    # Select an image file
    receipt_image_path = select_image()
    if not receipt_image_path:
        print("No file selected.")
        return

    # Process the image and extract text
    receipt_details = process_image(receipt_image_path)

    # Display the extracted text
    if receipt_details:
        print("\nExtracted Receipt Details:\n")
        print(receipt_details)

if __name__ == "__main__":
    main()

