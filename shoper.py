import cv2
import pytesseract

# Specify the path to your Tesseract executable
# Uncomment the line below and specify your Tesseract installation path if you are on Windows
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding for better OCR results
    _, binary_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Display processing steps (optional)
    cv2.imshow('Grayscale Image', gray)
    cv2.imshow('Binary Image', binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Extract text from the binarized image
    extracted_text = pytesseract.image_to_string(binary_image)

    return extracted_text

def main():
    # Path to the receipt image
    receipt_image_path = 'image/Recepts-5.png'  
    # Process the image and extRact text
    receipt_details = process_image(receipt_image_path)

    # Display the extracted details in the terminal
    print("Extracted Receipt Details:")
    print(receipt_details)

if __name__ == "__main__":
    main()
