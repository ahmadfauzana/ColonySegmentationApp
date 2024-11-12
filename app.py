from flask import Flask, render_template, request, redirect, url_for, send_file, flash, session
import os
import cv2
import random
import string
import numpy as np
import time

app = Flask(__name__)
app.secret_key = 'supersecretkey'
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'static/processed'
CODE_FOLDER = 'verification_codes'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(CODE_FOLDER, exist_ok=True)

def process_image(image_path):
    # Load the original image
    original = cv2.imread(image_path)
    
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Apply Gaussian Blurring to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Step 3: Enhance contrast with CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(blurred)
    
    # Step 4: Use adaptive thresholding to segment the image
    segmented = cv2.adaptiveThreshold(
        contrast_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Step 5: Morphological operations to refine segmentation
    kernel = np.ones((3, 3), np.uint8)
    # Remove small noise and close small holes in segmented regions
    segmented_refined = cv2.morphologyEx(segmented, cv2.MORPH_CLOSE, kernel)
    
    # Step 6: Edge detection using Canny for clear boundaries
    edges = cv2.Canny(contrast_enhanced, 50, 150)
    
    # Step 7: Convert segmented and edge images to BGR for color overlay
    segmented_bgr = cv2.cvtColor(segmented_refined, cv2.COLOR_GRAY2BGR)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Step 8: Combine the original image with segmented and edge-detected layers
    combined = cv2.addWeighted(original, 0.6, segmented_bgr, 0.3, 0)
    combined = cv2.addWeighted(combined, 0.9, edges_bgr, 0.3, 0)

    return original, segmented_bgr, combined

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

            original, segmented, combined = process_image(upload_path)
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
