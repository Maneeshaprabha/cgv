# Receipt OCR and Sales Visualization

This project is designed to extract text from receipt images using Optical Character Recognition (OCR) with Tesseract and visualize the parsed sales data using matplotlib.

## Features

- **OCR using Tesseract**: Extracts text from receipt images to identify items and their prices.
- **Image Preprocessing**: Enhances image quality for better OCR accuracy (adaptive thresholding, blurring, erosion, and dilation).
- **Sales Data Parsing**: Parses the extracted text into structured data (items and prices).
- **Data Visualization**: Visualizes the sales data using a horizontal bar chart.
- **GUI Interface**: Allows users to select images using a simple file dialog.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Tesseract OCR
- Tkinter (for the file dialog)
- Matplotlib (for visualization)
  
## Prerequisites

Before running the project, make sure you have the following software and libraries installed:

- **Python 3.x**: [Install Python](https://www.python.org/downloads/).
- **Tesseract OCR**: [Install Tesseract](https://github.com/tesseract-ocr/tesseract) for your platform.
- **Pip**: A package installer for Python (usually comes with Python).
- **Git**: (Optional) to clone this repository, [Install Git](https://git-scm.com/).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Maneeshaprabha/cgv.git
   cd your-repo-name
2. Download and install the latest version of Python from [https://www.python.org/downloads/](https://www.python.org/downloads/). Ensure you check the box to add Python to your system's PATH.


- **Windows**: Download and install Tesseract OCR from [Tesseract OCR for Windows](https://github.com/UB-Mannheim/tesseract/wiki).
- **macOS**: Install via Homebrew:
  ```bash
  brew install tesseract
