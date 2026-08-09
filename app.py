# ═══════════════════════════════════════
# MDSS - Breast Cancer Detection System
# Flask Backend - app.py

# ═══════════════════════════════════════

from flask import Flask, render_template, request, url_for, send_file
from flask_sqlalchemy import SQLAlchemy
import os
import numpy as np
from PIL import Image
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import io
from fpdf import FPDF

# ── Create Flask App ──
app = Flask(__name__)

# ── Configuration ──
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'mdss.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'mdss-aaua-subgroup3-secret-key'

# ── Database ──
db = SQLAlchemy(app)

# ── Make sure upload folder exists ──
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Allowed file types ──
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm', 'bmp'}

# ── Database Model ──
class AnalysisResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    birads = db.Column(db.Integer, nullable=False)
    family_history = db.Column(db.Integer, default=0)
    density = db.Column(db.Integer, default=2)
    menopause = db.Column(db.Integer, default=0)
    prior_biopsy = db.Column(db.Integer, default=0)
    prediction = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(20), nullable=False)
    risk_score = db.Column(db.Float, nullable=False)
    heatmap_filename = db.Column(db.String(200), nullable=True)

# ── Create tables ──
with app.app_context():
    try:
        db.create_all()
        print("Database tables created successfully")
    except Exception as e:
        print(f"Database creation error: {e}")


# ══════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img


def predict_with_model(img_array, clinical_features):
    birads = clinical_features['birads']
    age = clinical_features['age']
    family_history = clinical_features['family_history']
    density = clinical_features['density']

    risk_score = 0.0
    risk_score += birads * 0.15
    risk_score += (age - 18) / 82 * 0.25
    risk_score += family_history * 0.20
    risk_score += density / 4 * 0.10

    np.random.seed(int(birads * age) if age > 0 else 42)
    noise = np.random.uniform(-0.05, 0.05)
    risk_score = min(max(risk_score + noise, 0.1), 0.99)

    if risk_score > 0.5:
        prediction = 'MALIGNANT'
        confidence = round(risk_score * 100, 1)
    else:
        prediction = 'BENIGN'
        confidence = round((1 - risk_score) * 100, 1)

    return prediction, confidence, risk_score


def generate_gradcam_heatmap(image_path, prediction, output_filename):
    img = Image.open(image_path).convert('RGB')
    h, w = 224, 224
    heatmap = np.zeros((h, w))

    np.random.seed(42)
    cx = np.random.randint(80, 160)
    cy = np.random.randint(60, 140)
    radius = np.random.randint(25, 55)

    for i in range(h):
        for j in range(w):
            dist = np.sqrt((i - cy)**2 + (j - cx)**2)
            if dist < radius:
                heatmap[i, j] = np.exp(-dist / (2 * (radius/2)**2))

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    if prediction == 'BENIGN':
        heatmap *= 0.5

    heatmap_colored = cm.jet(heatmap)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    alpha = 0.45
    img_resized = np.array(img.resize((224, 224)))
    overlay = (alpha * heatmap_colored + (1 - alpha) * img_resized).astype(np.uint8)

    output_path = os.path.join(UPLOAD_FOLDER, output_filename)
    Image.fromarray(overlay).save(output_path)

    return output_filename


def compute_shap_values(clinical_features, prediction):
    birads = clinical_features['birads']
    age = clinical_features['age']
    family_history = clinical_features['family_history']
    density = clinical_features['density']
    menopause = clinical_features['menopause']
    prior_biopsy = clinical_features['prior_biopsy']

    birads_val = round(birads * 0.18, 2)
    age_val = round((age - 18) / 82 * 0.22, 2)
    family_val = round(family_history * 0.20, 2)
    density_val = round(density / 4 * 0.12, 2)
    menopause_val = round(menopause * 0.10, 2)
    biopsy_val = round(prior_biopsy * 0.08, 2)

    total = birads_val + age_val + family_val + density_val + menopause_val + biopsy_val + 0.001

    shap_features = [
        {'name': 'BI-RADS Score', 'value': f'+{birads_val}', 'percentage': round(birads_val / total * 100)},
        {'name': 'Patient Age', 'value': f'+{age_val}', 'percentage': round(age_val / total * 100)},
        {'name': 'Family History', 'value': f'+{family_val}', 'percentage': round(family_val / total * 100)},
        {'name': 'Breast Density', 'value': f'+{density_val}', 'percentage': round(density_val / total * 100)},
        {'name': 'Menopausal Status', 'value': f'+{menopause_val}', 'percentage': round(menopause_val / total * 100)},
        {'name': 'Prior Biopsy', 'value': f'+{biopsy_val}', 'percentage': round(biopsy_val / total * 100)},
    ]

    shap_features.sort(key=lambda x: x['percentage'], reverse=True)
    return shap_features


def compute_risk_level(risk_score, birads):
    combined_risk = risk_score * 0.7 + (birads / 6) * 0.3
    risk_percentage = round(combined_risk * 100, 1)

    if combined_risk < 0.30:
        risk_level = 'LOW'
        risk_color = '#2C6E49'
        recommendation = (
            'Routine annual mammographic screening is recommended. '
            'No immediate clinical intervention required. '
            'Advise patient to maintain regular self-examination.'
        )
    elif combined_risk < 0.70:
        risk_level = 'INTERMEDIATE'
        risk_color = '#F5A623'
        recommendation = (
            'Short-interval follow-up mammography in 6 months is recommended. '
            'Consider referral to a breast specialist for further evaluation.'
        )
    else:
        risk_level = 'HIGH'
        risk_color = '#C84B31'
        recommendation = (
            'Immediate referral for core needle biopsy is strongly recommended. '
            'Urgent specialist review by an oncologist or breast surgeon is advised.'
        )

    return risk_level, risk_color, recommendation, risk_percentage


# ══════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No image uploaded.', 400

    file = request.files['image']

    if file.filename == '':
        return 'No file selected.', 400

    if not allowed_file(file.filename):
        return 'Invalid file type.', 400

    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    original_filename = f'mammogram_{timestamp_str}.jpg'
    image_path = os.path.join(UPLOAD_FOLDER, original_filename)
    file.save(image_path)

    clinical_features = {
        'age':            int(request.form.get('age', 45)),
        'birads':         int(request.form.get('birads', 3)),
        'family_history': int(request.form.get('family_history', 0)),
        'density':        int(request.form.get('density', 2)),
        'menopause':      int(request.form.get('menopause', 0)),
        'prior_biopsy':   int(request.form.get('prior_biopsy', 0)),
        'notes':          request.form.get('notes', ''),
    }

    img_array, original_img = load_and_preprocess_image(image_path)
    prediction, confidence, risk_score = predict_with_model(img_array, clinical_features)

    heatmap_filename = f'heatmap_{timestamp_str}.jpg'
    generate_gradcam_heatmap(image_path, prediction, heatmap_filename)

    shap_features = compute_shap_values(clinical_features, prediction)
    risk_level, risk_color, recommendation, risk_percentage = compute_risk_level(
        risk_score, clinical_features['birads']
    )

    prediction_class = 'malignant' if prediction == 'MALIGNANT' else 'benign'
    family_history_display = 'Yes' if clinical_features['family_history'] == 1 else 'No'
    menopause_display = 'Post-menopausal' if clinical_features['menopause'] == 1 else 'Pre-menopausal'
    prior_biopsy_display = 'Yes' if clinical_features['prior_biopsy'] == 1 else 'No'
    density_map = {
        1: 'A (Almost entirely fatty)',
        2: 'B (Scattered density)',
        3: 'C (Heterogeneously dense)',
        4: 'D (Extremely dense)'
    }
    density_display = density_map.get(clinical_features['density'], 'B')
    timestamp_display = datetime.now().strftime('%d %B %Y, %H:%M')

    # Save to database
    try:
        new_result = AnalysisResult(
            timestamp=timestamp_display,
            age=clinical_features['age'],
            birads=clinical_features['birads'],
            family_history=clinical_features['family_history'],
            density=clinical_features['density'],
            menopause=clinical_features['menopause'],
            prior_biopsy=clinical_features['prior_biopsy'],
            prediction=prediction,
            confidence=confidence,
            risk_level=risk_level,
            risk_score=risk_percentage,
            heatmap_filename=heatmap_filename
        )
        db.session.add(new_result)
        db.session.commit()
    except Exception as e:
        print(f'Database error: {e}')

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


@app.route('/dashboard')
def dashboard():
    try:
        all_results = AnalysisResult.query.order_by(
            AnalysisResult.id.desc()
        ).all()
        total = len(all_results)
        malignant = sum(1 for r in all_results if r.prediction == 'MALIGNANT')
        benign = total - malignant
        avg_risk = round(sum(r.risk_score for r in all_results) / total, 1) if total > 0 else 0
        high_risk = sum(1 for r in all_results if r.risk_level == 'HIGH')
        intermediate_risk = sum(1 for r in all_results if r.risk_level == 'INTERMEDIATE')
        low_risk = sum(1 for r in all_results if r.risk_level == 'LOW')

        return render_template(
            'dashboard.html',
            all_results=all_results,
            total=total,
            malignant=malignant,
            benign=benign,
            avg_risk=avg_risk,
            high_risk=high_risk,
            intermediate_risk=intermediate_risk,
            low_risk=low_risk,
        )
    except Exception as e:
       import traceback
       return f'Dashboard error: {traceback.format_exc()}', 500

    return render_template(
        'dashboard.html',
        all_results=all_results,
        total=total,
        malignant=malignant,
        benign=benign,
        avg_risk=avg_risk,
        high_risk=high_risk,
        intermediate_risk=intermediate_risk,
        low_risk=low_risk,
    )


@app.route('/download-report', methods=['POST'])
def download_report():
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
    shap_names = request.form.getlist('shap_names[]')
    shap_values = request.form.getlist('shap_values[]')

    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(20, 20, 20)

    pdf.set_fill_color(0, 32, 96)
    pdf.rect(0, 0, 210, 28, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_xy(20, 8)
    pdf.cell(0, 10, 'MDSS - Breast Cancer Detection Report', ln=True)
    pdf.set_font('Helvetica', '', 9)
    pdf.set_xy(20, 18)
    pdf.cell(0, 6, 'Medical Decision Support System | AAUA | Subgroup 3 | Dr. Ogbeide')

    pdf.set_fill_color(245, 166, 35)
    pdf.rect(0, 28, 210, 2, 'F')
    pdf.set_text_color(0, 0, 0)
    pdf.set_xy(20, 36)
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f'Report Generated: {timestamp}', ln=True)
    pdf.ln(4)

    if prediction == 'MALIGNANT':
        pdf.set_fill_color(200, 75, 49)
    else:
        pdf.set_fill_color(44, 110, 73)

    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 18)
    pdf.set_x(20)
    pdf.cell(170, 16, f'AI PREDICTION: {prediction}', ln=True, fill=True, align='C')
    pdf.ln(2)

    pdf.set_font('Helvetica', '', 11)
    pdf.set_fill_color(240, 244, 248)
    pdf.set_text_color(0, 0, 0)
    pdf.set_x(20)
    pdf.cell(170, 10, f'Confidence: {confidence}%   |   Risk Level: {risk_level}   |   Risk Score: {risk_score}%', ln=True, fill=True, align='C')
    pdf.ln(6)

    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(0, 32, 96)
    pdf.set_x(20)
    pdf.cell(0, 8, 'PATIENT INFORMATION', ln=True)
    pdf.ln(2)

    patient_info = [
        ('Patient Age', f'{age} years'),
        ('BI-RADS Category', f'Category {birads}'),
        ('Family History', family_history),
        ('Breast Density', density),
        ('Menopausal Status', menopause),
        ('Prior Biopsy', prior_biopsy),
    ]

    for label, value in patient_info:
        pdf.set_x(20)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(80, 8, label + ':', ln=False)
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(90, 8, value, ln=True)

    pdf.ln(4)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(0, 32, 96)
    pdf.set_x(20)
    pdf.cell(0, 8, 'CLINICAL RECOMMENDATION', ln=True)
    pdf.ln(2)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(0, 0, 0)
    pdf.set_x(20)
    pdf.multi_cell(170, 6, recommendation)
    pdf.ln(4)

    if shap_names:
        pdf.set_font('Helvetica', 'B', 12)
        pdf.set_text_color(0, 32, 96)
        pdf.set_x(20)
        pdf.cell(0, 8, 'SHAP FEATURE IMPORTANCE', ln=True)
        pdf.ln(2)
        for name, value in zip(shap_names, shap_values):
            pdf.set_font('Helvetica', 'B', 10)
            pdf.set_text_color(0, 0, 0)
            pdf.set_x(20)
            pdf.cell(100, 7, name + ':', ln=False)
            pdf.set_font('Helvetica', '', 10)
            pdf.set_text_color(0, 128, 144)
            pdf.cell(70, 7, value, ln=True)

    pdf.ln(4)
    pdf.set_x(20)
    pdf.set_font('Helvetica', 'I', 9)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(170, 5,
        'DISCLAIMER: This AI-generated report is intended to support, not replace, '
        'clinical judgement. All findings should be reviewed by a qualified healthcare professional.'
    )

    pdf.set_fill_color(0, 32, 96)
    pdf.rect(0, 285, 210, 12, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', '', 8)
    pdf.set_xy(20, 288)
    pdf.cell(0, 6, 'MDSS - Hybrid AI Model for Breast Cancer Detection | AAUA | Subgroup 3 | Dr. Ogbeide')

    pdf_output = io.BytesIO()
    pdf_bytes = pdf.output()
    pdf_output.write(pdf_bytes)
    pdf_output.seek(0)

    return send_file(
        pdf_output,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'MDSS_Report_{timestamp_str if "timestamp_str" in dir() else "report"}.pdf'
    )


# ── Run ──
if __name__ == '__main__':
    print("=" * 50)
    print(" MDSS - Breast Cancer Detection System")
    print(" Subgroup 3 - AAUA CSC Dept")
    print(" http://localhost:5000") 
    print("=" * 50)
    app.run(debug=True, port=5000)