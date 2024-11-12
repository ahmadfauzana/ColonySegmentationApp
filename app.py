from flask import Flask, render_template, request, redirect, url_for, send_file, flash, session
import os
import cv2
import random
import string
import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans

app = Flask(__name__)
app.secret_key = 'supersecretkey'
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'static/processed'
CODE_FOLDER = 'verification_codes'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(CODE_FOLDER, exist_ok=True)

def segment_colonies(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding to create a binary image
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological opening to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Distance transform to identify colony centers
    dist_transform = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    dist_transform = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Threshold for markers for Watershed
    _, markers = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    markers = ndimage.label(markers)[0]
    
    # Apply the Watershed algorithm
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]  # Boundary in red
    
    # Color quantization using KMeans for a colorful, scattered effect
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    kmeans = KMeans(n_clusters=5, random_state=0).fit(pixel_values)
    centers = np.uint8(kmeans.cluster_centers_)
    segmented_image = centers[kmeans.labels_]
    segmented_image = segmented_image.reshape(image.shape)
    
    # Mask areas outside the segmented colonies
    segmented_mask = np.zeros_like(segmented_image)
    segmented_mask[markers > 1] = segmented_image[markers > 1]
    
    # Combine original image with segmented output for an overlay effect
    combined = cv2.addWeighted(image, 0.5, segmented_mask, 0.5, 0)
    
    # Return images
    return image, segmented_mask, combined

def generate_captcha_code(length=6):
    # Generate a random alphanumeric string for the CAPTCHA
    characters = string.ascii_uppercase + string.digits
    return ''.join(random.choices(characters, k=length))

def save_verification_code(code, filename):
    code_path = os.path.join(CODE_FOLDER, f'verification_{filename}.txt')
    with open(code_path, 'w') as f:
        f.write(code)
    return code_path

def read_verification_code(filename):
    code_path = os.path.join(CODE_FOLDER, f'verification_{filename}.txt')
    if os.path.exists(code_path):
        with open(code_path, 'r') as f:
            return f.read()
    return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        verification_code_input = request.form['verification_code']
        if file:
            filename = file.filename
            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(upload_path)

            original, segmented, combined = segment_colonies(upload_path)
            new_verification_code = generate_captcha_code()

            saved_code = read_verification_code(filename)
            if saved_code:
                if not verification_code_input:
                    flash('Error: This image has been uploaded before. Please provide the correct verification code.')
                    return redirect(url_for('index'))
                if verification_code_input != saved_code:
                    flash('Error: The verification code you provided does NOT match the uploaded image.')
                    return redirect(url_for('index'))
            else:
                save_verification_code(new_verification_code, filename)

            original_path = os.path.join(PROCESSED_FOLDER, f'original_{filename}')
            segmented_path = os.path.join(PROCESSED_FOLDER, f'segmented_{filename}')
            combined_path = os.path.join(PROCESSED_FOLDER, f'combined_{filename}')

            cv2.imwrite(original_path, original)
            cv2.imwrite(segmented_path, segmented)
            cv2.imwrite(combined_path, combined)

            return redirect(url_for('results', filename=filename))
        
    return render_template('index.html')

@app.route('/results/<filename>')
def results(filename):
    original_url = url_for('static', filename=f'processed/original_{filename}')
    segmented_url = url_for('static', filename=f'processed/segmented_{filename}')
    combined_url = url_for('static', filename=f'processed/combined_{filename}')
    verification_code = read_verification_code(filename)
    return render_template('results.html', original_url=original_url, segmented_url=segmented_url, combined_url=combined_url, verification_code=verification_code)

if __name__ == "__main__":
    app.run(debug=True)
