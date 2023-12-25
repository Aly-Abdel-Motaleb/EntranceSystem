# Entrance System with License Plate Detection and Recognition

## Project Overview

This project implements an entrance system that detects license plates from images and recognizes the characters on the plates using Optical Character Recognition (OCR) with logistic regression. The system is designed to be integrated with a web application, allowing users to upload images and receive the detected license plate along with the recognized characters.

## Project Structure

The project is organized into the following components:

1. **License Plate Detection (LPD):**
   - The LPD module is responsible for detecting license plates in the input image.
   - It uses edge detection, morphological operations, and contour analysis to identify potential license plate regions.
   - The LPD function is named `LPD(img)` and returns the cropped license plate image.

2. **Character Recognition:**
   - The character recognition module extracts individual characters from the license plate and recognizes them.
   - It uses various image processing techniques, including contour analysis and HOG (Histogram of Oriented Gradients) feature extraction.
   - The character recognition function is named `extractChars(img)` and returns the processed license plate image along with the recognized characters.

3. **Character Recognition with OCR:**
   - The character recognition module extracts individual characters from the license plate and recognizes them using Optical Character Recognition (OCR) with logistic regression.
   - It uses various image processing techniques, including contour analysis and HOG (Histogram of Oriented Gradients) feature extraction.
   - The character recognition function is named `extractChars(img)` and returns the processed license plate image along with the recognized characters.

4. **Web Application (Flask Server):**
   - The Flask server (`server.py`) provides the backend for the web application.
   - The server handles image uploads, processes images using the LPD and OCR modules, and returns the results to the frontend.

## Required Libraries

Ensure you have the following libraries installed before running the project:

- OpenCV (`cv2`)
- Imutils
- NumPy
- Pandas
- Joblib
- scikit-image (`skimage`)
- Flask

Install the required libraries using the following command:

```bash
pip install opencv-python imutils numpy pandas joblib scikit-image Flask
```

## How to Run

1. Clone the repository or download the provided code files.
2. Install the required libraries as mentioned above.
3. Run the Flask server using the following command:

```bash
python server.py
```

4. Access the web application by visiting [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your web browser.

5. Upload an image using the provided interface.

6. The system will process the image, detect the license plate, recognize characters using OCR with logistic regression, and display the result.

## Project Files

- **`Detector.py`:** Contains the implementation of the LPD and OCR modules.
- **`server.py`:** Implements the Flask server for the web application.
- **`templates/index.html`:** HTML template for the web application interface.

## Important Notes

- The LPD and character recognition modules are implemented in the `Detector.py` file.
- The logistic regression OCR model is loaded using the `load` function from the `joblib` library.
- The `server.py` file contains the Flask server code for running the web application.
- The web application provides an interface for uploading images and displays the processed result.

**Note:** Ensure that you have the necessary permissions to use the trained model and any proprietary code or data included in this project.