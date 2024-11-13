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
    # Load the original grayscale image
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Step 1: Enhance contrast with CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray)
    
    # Step 2: Apply a color map to the contrast-enhanced grayscale image for pseudo-coloring
    pseudo_color = cv2.applyColorMap(contrast_enhanced, cv2.COLORMAP_JET)
    
    # Step 3: Adaptive thresholding to isolate colony clusters
    _, thresholded = cv2.threshold(contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Step 4: Morphological operations to clean up small noise and close gaps
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Step 5: Find contours to isolate clusters and improve boundary details
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Step 6: Detect circles using HoughCircles
    circles = cv2.HoughCircles(contrast_enhanced, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=100, param2=30, minRadius=10, maxRadius=100)
    
    # If some circles are detected, proceed with clustering
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        # Step 7: Create an image to hold the colored colonies within the detected circles
        colony_colored_image = pseudo_color.copy()
        
        for i, (x, y, r) in enumerate(circles):
            # Create a mask for the current circle
            mask = np.zeros_like(gray)
            cv2.circle(mask, (x, y), r, 255, thickness=cv2.FILLED)
            
            # Apply the mask to isolate colonies inside the circle
            isolated_colony = cv2.bitwise_and(morph, morph, mask=mask)
            
            # Find contours inside the circle to identify individual colonies
            colony_contours, _ = cv2.findContours(isolated_colony, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Generate random colors for each colony (in BGR format)
            colony_color = np.random.randint(0, 255, size=(1, 3), dtype=np.uint8).tolist()[0]
            
            # Draw the contours of the colonies inside the circle with unique colors
            for contour in colony_contours:
                cv2.drawContours(colony_colored_image, [contour], -1, colony_color, 2)
        
        # Step 8: Final combined image - blend pseudo-color and highlighted colonies with darkened background
        final_combined = cv2.addWeighted(colony_colored_image, 0.8, pseudo_color, 0.2, 0)
        
    else:
        # If no circles are detected, use the original pseudo-colored image
        final_combined = pseudo_color

    return pseudo_color, colony_colored_image, final_combined

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
