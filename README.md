# Receipt OCR and Sales Visualization

This project is designed to extract text from receipt images using Optical Character Recognition (OCR) with Tesseract and visualize the parsed sales data using matplotlib.

## Features

- **OCR using Tesseract**: Extracts text from receipt images to identify items and their prices.
- **Image Preprocessing**: Enhances image quality for better OCR accuracy (adaptive thresholding, blurring, erosion, and dilation).
- **Sales Data Parsing**: Parses the extracted text into structured data (items and prices).
- **Data Visualization**: Visualizes the sales data using a horizontal bar chart.
- **GUI Interface**: Allows users to select images using a simple file dialog.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Maneeshaprabha/cgv
   cd your-repo-name
