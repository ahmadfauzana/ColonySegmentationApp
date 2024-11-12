# Colony Segmentation App

This project is a tool that allows users to upload images, perform segmentation to recognize individual colonies, and view or download the processed images. The application generates a unique verification code for each image, ensuring that outputs can be correctly linked to their respective inputs.

## Features

- Converts grayscale images to binary images for segmentation.
- Applies morphological operations to remove noise.
- Uses the Watershed algorithm to segment colony-like regions.
- Creates a colorful, scattered effect for easier visualization of colonies.
- Outputs three images: the original, segmented, and combined overlay.

## Prerequisites

Ensure you have the following libraries installed:

- Python 3.9
- Flask
- OpenCV
- NumPy
- SciPy
- Matplotlib
- Scikit-Learn (for KMeans clustering)

To install these dependencies, you can use:

```bash
pip install flask opencv-python numpy scipy matplotlib

```

## Usage

```bash
git clone https://github.com/ahmadfauzana/colonysegmentationapp.git
cd colonysegmentationapp
```

## Project Structure

├── app.py                  # Main Flask application
├── uploads                 # Folder for uploaded images
├── output_images           # Folder for saving processed images
└── templates               # HTML templates for the web interface
    ├── index.html          # Upload form
    └── result.html         # Display results and download links

## Run & Access the Application

```bash
python app.py
```

Open your web browser and navigate to `http://127.0.0.1:5000`. You can upload an image in the web browser.
The application will process the image and display the results.

1. Upload an image in the web browser.
   1. On the homepage, click Choose File to select an image in .png, .jpg, or .jpeg format, then click Upload.
2. View Results
   1. After uploading, the app will display the original, segmented, and combined images
   2. It will also generate a verification code for the uploaded image.
3. Download Results
   1. Files available that already downloaded:
      1. Original Image
      2. Segmented Image
      3. Combined Image
      4. Verification Code

## Screenshots

1. Homepage
   ![Homepage](<results\homepage.png>)
2. Results Page
   ![Results Page](<results\homepage.png>)

## License

No license on this project.
