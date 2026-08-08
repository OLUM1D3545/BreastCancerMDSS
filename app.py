# ═══════════════════════════════════════
# MDSS - Breast Cancer Detection System
# Flask Backend - app.py
# ═══════════════════════════════════════

# ── Imports ──
# Flask: the web framework that runs the website
from flask import Flask, render_template, request, url_for
# OS: helps us work with files and folders
import os
# NumPy: for numerical computations
import numpy as np
# Pillow: for opening and processing images
from PIL import Image
# DateTime: for timestamping the reports
from datetime import datetime
# Matplotlib: for drawing the Grad-CAM heatmap
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from fpdf import FPDF
import io
from flask import send_file

# ── Create Flask App ──
app = Flask(__name__)

# ── Configuration ──
# This tells Flask where to save uploaded mammogram images
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Make sure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Allowed File Types ──
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm', 'bmp'}

def allowed_file(filename):
    """Check if the uploaded file is an allowed image type"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ══════════════════════════════════════════════
# AI MODEL FUNCTIONS
# ══════════════════════════════════════════════

def load_and_preprocess_image(image_path):
    """
    Load and preprocess the mammogram image for the CNN.
    Resize to 224x224 (VGG16 requirement) and normalise.
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalise to 0-1
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array, img


def predict_with_model(img_array, clinical_features):
    """
    Run the hybrid CNN-SVM prediction.

    NOTE: This is a DEMO version. When the real trained model
    is ready from Google Colab, replace this function with
    the actual model prediction code.

    For the demo, we simulate a realistic prediction based
    on the clinical features provided.
    """

    # ── DEMO PREDICTION LOGIC ──
    # This simulates what the real CNN-SVM model would do.
    # It uses the clinical features to generate a realistic
    # prediction for demonstration purposes.

    birads = clinical_features['birads']
    age = clinical_features['age']
    family_history = clinical_features['family_history']
    density = clinical_features['density'] 

    # Simple risk scoring based on clinical factors
    # (In the real model, this would come from the CNN+SVM)
    risk_score = 0.0
    risk_score += birads * 0.15        # BI-RADS contributes most
    risk_score += (age - 18) / 82 * 0.25  # Age factor
    risk_score += family_history * 0.20   # Family history
    risk_score += density / 4 * 0.10      # Breast density

    # Add some realistic variation
    np.random.seed(int(birads * age))
    noise = np.random.uniform(-0.05, 0.05)
    risk_score = min(max(risk_score + noise, 0.1), 0.99)

    # Determine prediction
    if risk_score > 0.5:
        prediction = 'MALIGNANT'
        confidence = round(risk_score * 100, 1)
    else:
        prediction = 'BENIGN'
        confidence = round((1 - risk_score) * 100, 1)

    return prediction, confidence, risk_score


def generate_gradcam_heatmap(image_path, prediction, output_filename):
    """
    Generate a Grad-CAM style heatmap overlaid on the mammogram.

    NOTE: This is a DEMO version that creates a realistic-looking
    heatmap. When the real CNN model is integrated, replace this
    with actual Grad-CAM computation using TensorFlow GradientTape.
    """
    # Load original image
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img.resize((224, 224)))

    # Create a simulated activation map
    # In the real model this comes from the CNN's last conv layer
    h, w = 224, 224
    heatmap = np.zeros((h, w))

    # Simulate suspicious region in centre-upper area
    # (common location for breast masses)
    np.random.seed(42)
    cx = np.random.randint(80, 160)
    cy = np.random.randint(60, 140)
    radius = np.random.randint(25, 55)

    for i in range(h):
        for j in range(w):
            dist = np.sqrt((i - cy)**2 + (j - cx)**2)
            if dist < radius:
                heatmap[i, j] = np.exp(-dist / (2 * (radius/2)**2))

    # Add some secondary activations
    cx2 = cx + np.random.randint(-40, 40)
    cy2 = cy + np.random.randint(-30, 30)
    radius2 = radius // 2
    for i in range(h):
        for j in range(w):
            dist = np.sqrt((i - cy2)**2 + (j - cx2)**2)
            if dist < radius2:
                heatmap[i, j] += 0.4 * np.exp(-dist / (2 * (radius2/2)**2))

    # Normalise heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # If benign, make the heatmap less intense
    if prediction == 'BENIGN':
        heatmap *= 0.5

    # Apply colormap (jet: blue=low, red=high activation)
    heatmap_colored = cm.jet(heatmap)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # Overlay heatmap on original image
    alpha = 0.45
    img_resized = np.array(img.resize((224, 224)))
    overlay = (alpha * heatmap_colored + (1 - alpha) * img_resized).astype(np.uint8)

    # Save the heatmap image
    output_path = os.path.join(UPLOAD_FOLDER, output_filename)
    Image.fromarray(overlay).save(output_path)

    return output_filename


def compute_shap_values(clinical_features, prediction):
    """
    Compute SHAP feature importance values.

    NOTE: This is a DEMO version with realistic SHAP-style outputs.
    When the real model is integrated, replace with actual SHAP
    library computation using shap.TreeExplainer or shap.KernelExplainer.
    """
    birads = clinical_features['birads']
    age = clinical_features['age']
    family_history = clinical_features['family_history']
    density = clinical_features['density']
    menopause = clinical_features['menopause']
    prior_biopsy = clinical_features['prior_biopsy']

    # Compute realistic feature contributions
    # Based on clinical literature on breast cancer risk factors
    birads_val = round(birads * 0.18, 2)
    age_val = round((age - 18) / 82 * 0.22, 2)
    family_val = round(family_history * 0.20, 2)
    density_val = round(density / 4 * 0.12, 2)
    menopause_val = round(menopause * 0.10, 2)
    biopsy_val = round(prior_biopsy * 0.08, 2)

    total = birads_val + age_val + family_val + density_val + menopause_val + biopsy_val + 0.001

    # Convert to percentages for the bar chart
    shap_features = [
        {
            'name': 'BI-RADS Score',
            'value': f'+{birads_val}',
            'percentage': round(birads_val / total * 100)
        },
        {
            'name': 'Patient Age',
            'value': f'+{age_val}',
            'percentage': round(age_val / total * 100)
        },
        {
            'name': 'Family History',
            'value': f'+{family_val}',
            'percentage': round(family_val / total * 100)
        },
        {
            'name': 'Breast Density',
            'value': f'+{density_val}',
            'percentage': round(density_val / total * 100)
        },
        {
            'name': 'Menopausal Status',
            'value': f'+{menopause_val}',
            'percentage': round(menopause_val / total * 100)
        },
        {
            'name': 'Prior Biopsy',
            'value': f'+{biopsy_val}',
            'percentage': round(biopsy_val / total * 100)
        },
    ]

    # Sort by contribution (highest first)
    shap_features.sort(key=lambda x: x['percentage'], reverse=True)

    return shap_features


def compute_risk_level(risk_score, birads):
    """
    Compute the final risk tier and clinical recommendation
    based on the model risk score and BI-RADS category.
    """
    # Combine model risk score with BI-RADS
    combined_risk = risk_score * 0.7 + (birads / 6) * 0.3
    risk_percentage = round(combined_risk * 100, 1)

    if combined_risk < 0.30:
        risk_level = 'LOW'
        risk_color = '#2C6E49'
        recommendation = (
            'Routine annual mammographic screening is recommended. '
            'No immediate clinical intervention required. '
            'Advise patient to maintain regular self-examination and attend scheduled screenings.'
        )
    elif combined_risk < 0.70:
        risk_level = 'INTERMEDIATE'
        risk_color = '#F5A623'
        recommendation = (
            'Short-interval follow-up mammography in 6 months is recommended. '
            'Consider referral to a breast specialist for further clinical evaluation. '
            'Patient should be advised to report any changes in symptoms immediately.'
        )
    else:
        risk_level = 'HIGH'
        risk_color = '#C84B31'
        recommendation = (
            'Immediate referral for core needle biopsy is strongly recommended. '
            'Urgent specialist review by an oncologist or breast surgeon is advised. '
            'Do not delay further diagnostic workup. Patient counselling required.'
        )

    return risk_level, risk_color, recommendation, risk_percentage


# ══════════════════════════════════════════════
# FLASK ROUTES
# ══════════════════════════════════════════════

@app.route('/')
def index():
    """
    Home page — shows the upload form and clinical data input.
    When the doctor visits the website, this is the first page they see.
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    This function runs when the doctor clicks Run Analysis.
    It receives the uploaded image and clinical data,
    runs the AI model, and returns the results page.
    """

    # ── Step 1: Get the uploaded image ──
    if 'image' not in request.files:
        return 'No image uploaded. Please go back and upload a mammogram.', 400

    file = request.files['image']

    if file.filename == '':
        return 'No file selected. Please go back and select a mammogram image.', 400

    if not allowed_file(file.filename):
        return 'Invalid file type. Please upload a JPEG, PNG, or DICOM image.', 400

    # Save the uploaded image
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    original_filename = f'mammogram_{timestamp_str}.jpg'
    image_path = os.path.join(UPLOAD_FOLDER, original_filename)
    file.save(image_path)

    # ── Step 2: Get the clinical data from the form ──
    clinical_features = {
        'age':            int(request.form.get('age', 45)),
        'birads':         int(request.form.get('birads', 3)),
        'family_history': int(request.form.get('family_history', 0)),
        'density':        int(request.form.get('density', 2)),
        'menopause':      int(request.form.get('menopause', 0)),
        'prior_biopsy':   int(request.form.get('prior_biopsy', 0)),
        'notes':          request.form.get('notes', ''),
    }

    # ── Step 3: Preprocess the image ──
    img_array, original_img = load_and_preprocess_image(image_path)

    # ── Step 4: Run the AI prediction ──
    prediction, confidence, risk_score = predict_with_model(
        img_array, clinical_features
    )

    # ── Step 5: Generate Grad-CAM heatmap ──
    heatmap_filename = f'heatmap_{timestamp_str}.jpg'
    generate_gradcam_heatmap(image_path, prediction, heatmap_filename)

    # ── Step 6: Compute SHAP values ──
    shap_features = compute_shap_values(clinical_features, prediction)

    # ── Step 7: Compute risk level and recommendation ──
    risk_level, risk_color, recommendation, risk_percentage = compute_risk_level(
        risk_score, clinical_features['birads']
    )

    # ── Step 8: Prepare display values ──
    prediction_class = 'malignant' if prediction == 'MALIGNANT' else 'benign'

    family_history_display = 'Yes' if clinical_features['family_history'] == 1 else 'No'
    menopause_display = 'Post-menopausal' if clinical_features['menopause'] == 1 else 'Pre-menopausal'
    prior_biopsy_display = 'Yes' if clinical_features['prior_biopsy'] == 1 else 'No'
    density_map = {1: 'A (Almost entirely fatty)', 2: 'B (Scattered density)', 3: 'C (Heterogeneously dense)', 4: 'D (Extremely dense)'}
    density_display = density_map.get(clinical_features['density'], 'B')

    timestamp_display = datetime.now().strftime('%d %B %Y, %H:%M')

    # ── Step 9: Send everything to the result page ──
    return render_template(
        'result.html',
        prediction=prediction,
        prediction_class=prediction_class,
        confidence=confidence,
        risk_level=risk_level,
        risk_color=risk_color,
        risk_score=risk_percentage,
        recommendation=recommendation,
        heatmap_filename=heatmap_filename,
        shap_features=shap_features,
        age=clinical_features['age'],
        birads=clinical_features['birads'],
        family_history=family_history_display,
        density=density_display,
        menopause=menopause_display,
        prior_biopsy=prior_biopsy_display,
        timestamp=timestamp_display,
    )

# ── PDF Report Generation ──
@app.route('/download-report', methods=['POST'])
def download_report():
    """
    Generates a professional PDF clinical report
    and sends it to the doctor for download.
    """

    # Get all the result data from the form
    prediction = request.form.get('prediction', 'UNKNOWN')
    confidence = request.form.get('confidence', '0')
    risk_level = request.form.get('risk_level', 'UNKNOWN')
    risk_score = request.form.get('risk_score', '0')
    recommendation = request.form.get('recommendation', '')
    age = request.form.get('age', '')
    birads = request.form.get('birads', '')
    family_history = request.form.get('family_history', '')
    density = request.form.get('density', '')
    menopause = request.form.get('menopause', '')
    prior_biopsy = request.form.get('prior_biopsy', '')
    timestamp = request.form.get('timestamp', '')

    # SHAP features
    shap_names = request.form.getlist('shap_names[]')
    shap_values = request.form.getlist('shap_values[]')

    # ── Build the PDF ──
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(20, 20, 20)

    # Header bar
    pdf.set_fill_color(0, 32, 96)  # Navy
    pdf.rect(0, 0, 210, 28, 'F')

    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_xy(20, 8)
    pdf.cell(0, 10, 'MDSS - Breast Cancer Detection Report', ln=True)

    pdf.set_font('Helvetica', '', 9)
    pdf.set_xy(20, 18)
    pdf.cell(0, 6, 'Medical Decision Support System | AAUA Faculty of Computing | Subgroup 3 | Supervised by Dr. Ogbeide')

    # Gold line
    pdf.set_fill_color(245, 166, 35)  # Gold
    pdf.rect(0, 28, 210, 2, 'F')

    # Reset text color
    pdf.set_text_color(0, 0, 0)
    pdf.set_xy(20, 36)

    # Report date
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f'Report Generated: {timestamp}', ln=True)
    pdf.ln(4)

    # ── Prediction Result Box ──
    if prediction == 'MALIGNANT':
        pdf.set_fill_color(200, 75, 49)  # Red
    else:
        pdf.set_fill_color(44, 110, 73)  # Green

    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 18)
    pdf.set_x(20)
    pdf.cell(170, 16, f'AI PREDICTION: {prediction}', ln=True, fill=True, align='C')
    pdf.ln(2)

    pdf.set_font('Helvetica', '', 11)
    pdf.set_fill_color(240, 244, 248)
    pdf.set_text_color(0, 0, 0)
    pdf.set_x(20)
    pdf.cell(170, 10, f'Model Confidence: {confidence}%   |   Risk Level: {risk_level}   |   Risk Score: {risk_score}%', ln=True, fill=True, align='C')
    pdf.ln(6)

    # ── Patient Information ──
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(0, 32, 96)
    pdf.set_x(20)
    pdf.cell(0, 8, 'PATIENT INFORMATION', ln=True)

    pdf.set_fill_color(0, 32, 96)
    pdf.rect(20, pdf.get_y(), 170, 0.5, 'F')
    pdf.ln(4)

    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(0, 0, 0)

    patient_info = [
        ('Patient Age', f'{age} years'),
        ('BI-RADS Category', f'Category {birads}'),
        ('Family History of Breast Cancer', family_history),
        ('Breast Density', density),
        ('Menopausal Status', menopause),
        ('Prior Biopsy', prior_biopsy),
    ]

    for label, value in patient_info:
        pdf.set_x(20)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(80, 8, label + ':', ln=False)
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(90, 8, value, ln=True)

    pdf.ln(4)

    # ── Risk Assessment ──
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(0, 32, 96)
    pdf.set_x(20)
    pdf.cell(0, 8, 'RISK ASSESSMENT', ln=True)

    pdf.set_fill_color(0, 32, 96)
    pdf.rect(20, pdf.get_y(), 170, 0.5, 'F')
    pdf.ln(4)

    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(0, 0, 0)
    pdf.set_x(20)
    pdf.cell(80, 8, 'Risk Level:', ln=False)
    pdf.set_font('Helvetica', 'B', 10)
    if risk_level == 'HIGH':
        pdf.set_text_color(200, 75, 49)
    elif risk_level == 'INTERMEDIATE':
        pdf.set_text_color(200, 130, 0)
    else:
        pdf.set_text_color(44, 110, 73)
    pdf.cell(90, 8, f'{risk_level} RISK ({risk_score}%)', ln=True)
    pdf.set_text_color(0, 0, 0)

    pdf.ln(2)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_x(20)
    pdf.cell(0, 6, 'Clinical Recommendation:', ln=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_x(20)
    pdf.multi_cell(170, 6, recommendation)
    pdf.ln(4)

    # ── SHAP Values ──
    if shap_names:
        pdf.set_font('Helvetica', 'B', 12)
        pdf.set_text_color(0, 32, 96)
        pdf.set_x(20)
        pdf.cell(0, 8, 'SHAP FEATURE IMPORTANCE', ln=True)

        pdf.set_fill_color(0, 32, 96)
        pdf.rect(20, pdf.get_y(), 170, 0.5, 'F')
        pdf.ln(4)

        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(100, 100, 100)
        pdf.set_x(20)
        pdf.cell(0, 6, 'The following shows how much each clinical factor contributed to the AI prediction:', ln=True)
        pdf.ln(2)

        for name, value in zip(shap_names, shap_values):
            pdf.set_font('Helvetica', 'B', 10)
            pdf.set_text_color(0, 0, 0)
            pdf.set_x(20)
            pdf.cell(100, 7, name + ':', ln=False)
            pdf.set_font('Helvetica', '', 10)
            pdf.set_text_color(0, 128, 144)
            pdf.cell(70, 7, value, ln=True)
            pdf.set_text_color(0, 0, 0)

    pdf.ln(4)

    # ── Disclaimer ──
    pdf.set_fill_color(240, 244, 248)
    pdf.set_x(20)
    pdf.set_font('Helvetica', 'I', 9)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(170, 5,
        'DISCLAIMER: This AI-generated report is intended to support, not replace, clinical judgement. '
        'All findings should be reviewed by a qualified healthcare professional before any clinical decision is made. '
        'This system was developed by Subgroup 3, AAUA Faculty of Computing, under the supervision of Dr. Ogbeide.',
        fill=True
    )

    # ── Footer ──
    pdf.set_fill_color(0, 32, 96)
    pdf.rect(0, 285, 210, 12, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', '', 8)
    pdf.set_xy(20, 288)
    pdf.cell(0, 6, 'MDSS - Hybrid AI Model for Breast Cancer Detection | AAUA CSC Department | Subgroup 3 | Dr. Ogbeide')

    # ── Save and send PDF ──
    pdf_output = io.BytesIO()
    pdf_bytes = pdf.output()
    pdf_output.write(pdf_bytes)
    pdf_output.seek(0)

    return send_file(
        pdf_output,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'MDSS_Report_{timestamp.replace(" ", "_").replace(",", "")}.pdf'
    )
# ── Run the App ──
if __name__ == '__main__':
    print("=" * 50)
    print(" MDSS - Breast Cancer Detection System")
    print(" Developed by Subgroup 3 - AAUA CSC Dept")
    print(" Supervised by Dr. Ogbeide")
    print("=" * 50)
    print(" Open your browser and go to:")
    print(" http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
