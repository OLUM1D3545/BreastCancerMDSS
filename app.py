# ══════════════════════════════════════════════
# MDSS - Breast Cancer Detection System
# Flask Backend - app.py
# ══════════════════════════════════════════════
from flask import (
    Flask,
    render_template,
    abort,
    request,
    send_file,
    redirect,
    jsonify,
    url_for,
    flash
)
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user
)
from flask_bcrypt import Bcrypt
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv
import os
import hmac
import io
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
from fpdf import FPDF
import json
import joblib
import shap
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
# ══════════════════════════════════════════════
# CREATE FLASK APP
# ══════════════════════════════════════════════
load_dotenv()
app = Flask(__name__)
# ══════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════
UPLOAD_FOLDER = os.path.join("static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
basedir = os.path.abspath(os.path.dirname(__file__))
app.config["SQLALCHEMY_DATABASE_URI"] = (
    "sqlite:///" + os.path.join(basedir, "mdss.db")
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.secret_key = os.getenv("SECRET_KEY") or os.urandom(32).hex()

# Professional access key for registration. You can override this in .env.
# Keep this value server-side; never validate it only in browser JavaScript.
PROFESSIONAL_ACCESS_KEY = os.getenv(
    "PROFESSIONAL_ACCESS_KEY",
    "MDSS-PRO-2026"
).strip()
PROFESSIONAL_ROLES = {"radiologist", "doctor", "oncologist"}
VALID_ROLES = PROFESSIONAL_ROLES | {"patient"}
# Google OAuth configuration. Put credentials in the project-root .env file.
oauth = OAuth(app)
GOOGLE_OAUTH_ENABLED = bool(
    os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET")
)
if GOOGLE_OAUTH_ENABLED:
    oauth.register(
        name="google",
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )
# ══════════════════════════════════════════════
# TRAINED AI MODEL CONFIGURATION
# ══════════════════════════════════════════════
MODEL_FOLDER = os.path.join(basedir, "model")
MODEL_PATH = os.path.join(MODEL_FOLDER, "hybrid_svm.pkl")
SVM_SCALER_PATH = os.path.join(MODEL_FOLDER, "svm_scaler.pkl")
CLINICAL_ENCODER_PATH = os.path.join(MODEL_FOLDER, "clinical_encoder.pkl")
CLINICAL_SCALER_PATH = os.path.join(MODEL_FOLDER, "clinical_scaler.pkl")
MODEL_CONFIG_PATH = os.path.join(MODEL_FOLDER, "model_config.json")
SHAP_BACKGROUND_PATH = os.path.join(MODEL_FOLDER, "shap", "shap_background.npy")
# Load the trained SVM and preprocessing objects.
shap_background = None
try:
    hybrid_svm = joblib.load(MODEL_PATH)
    svm_scaler = joblib.load(SVM_SCALER_PATH)
    clinical_encoder = joblib.load(CLINICAL_ENCODER_PATH)
    clinical_scaler = joblib.load(CLINICAL_SCALER_PATH)
    with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as config_file:
        model_config = json.load(config_file)
    print("Real hybrid AI model loaded successfully.")
    print(f"Model expects {hybrid_svm.n_features_in_} features.")
    # Load the real SHAP background used for structured-feature explanations.
    shap_background = np.load(SHAP_BACKGROUND_PATH).astype(np.float32)
    if shap_background.ndim != 2 or shap_background.shape[1] != 9:
        raise ValueError(
            f"Expected SHAP background shape (n, 9), got {shap_background.shape}."
        )
    print(f"Real SHAP background loaded successfully: {shap_background.shape}")
except Exception as e:
    hybrid_svm = None
    svm_scaler = None
    clinical_encoder = None
    clinical_scaler = None
    model_config = {}
    shap_background = None
    print(f"WARNING: AI model could not be loaded: {e}")
# Load VGG16 once and use it as the frozen image feature extractor.
try:
    vgg16_base = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    vgg16_base.trainable = False
    vgg16_feature_extractor = tf.keras.Sequential([
        vgg16_base,
        tf.keras.layers.GlobalAveragePooling2D()
    ])
    # Grad-CAM uses the final convolutional feature map from VGG16.
    gradcam_model = tf.keras.Model(
        inputs=vgg16_base.input,
        outputs=[
            vgg16_base.get_layer("block5_conv3").output,
            vgg16_base.output
        ]
    )
    print("VGG16 feature extractor loaded successfully.")
except Exception as e:
    vgg16_base = None
    vgg16_feature_extractor = None
    gradcam_model = None
    print(f"WARNING: VGG16 could not be loaded: {e}")
# ══════════════════════════════════════════════
# DATABASE AND LOGIN CONFIGURATION
# ══════════════════════════════════════════════
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = (
    "Please login to access this page."
)
# ══════════════════════════════════════════════
# CREATE UPLOAD FOLDER
# ══════════════════════════════════════════════
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# ══════════════════════════════════════════════
# ALLOWED FILE TYPES
# ══════════════════════════════════════════════
ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "dcm",
    "bmp"
}
# ══════════════════════════════════════════════
# USER MODEL
# ══════════════════════════════════════════════
class User(UserMixin, db.Model):
    id = db.Column(
        db.Integer,
        primary_key=True
    )
    name = db.Column(
        db.String(100),
        nullable=False
    )
    email = db.Column(
        db.String(120),
        unique=True,
        nullable=False
    )
    password = db.Column(
        db.String(200),
        nullable=False
    )
    role = db.Column(
        db.String(20),
        nullable=False,
        default="patient"
    )
    created_at = db.Column(
        db.String(100),
        nullable=True
    )
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))
# ══════════════════════════════════════════════
# ANALYSIS RESULT MODEL
# ══════════════════════════════════════════════
class AnalysisResult(db.Model):
    id = db.Column(
        db.Integer,
        primary_key=True
    )
    timestamp = db.Column(
        db.String(100),
        nullable=False
    )
    age = db.Column(
        db.Integer,
        nullable=False
    )
    birads = db.Column(
        db.Integer,
        nullable=False
    )
    family_history = db.Column(
        db.Integer,
        default=0
    )
    density = db.Column(
        db.Integer,
        default=2
    )
    menopause = db.Column(
        db.Integer,
        default=0
    )
    prior_biopsy = db.Column(
        db.Integer,
        default=0
    )
    prediction = db.Column(
        db.String(20),
        nullable=False
    )
    confidence = db.Column(
        db.Float,
        nullable=False
    )
    risk_level = db.Column(
        db.String(20),
        nullable=False
    )
    risk_score = db.Column(
        db.Float,
        nullable=False
    )
    heatmap_filename = db.Column(
        db.String(200),
        nullable=True
    )
    # Persist the original uploaded image filename so historical analyses
    # remain traceable to the image used for the prediction.
    image_filename = db.Column(
        db.String(200),
        nullable=True
    )
    # Persist the trained-model structured inputs used for this analysis.
    subtlety = db.Column(db.Integer, nullable=True)
    abnormality_type = db.Column(db.String(50), nullable=True)
    image_view = db.Column(db.String(20), nullable=True)
    breast_side = db.Column(db.String(20), nullable=True)
    # The SVM malignant probability is distinct from the presentation-layer
    # MDSS Risk Index stored in risk_score.
    model_probability = db.Column(db.Float, nullable=True)
    # Store the local SHAP explanation so the historical analysis retains
    # the same explanation that was shown immediately after prediction.
    shap_data = db.Column(db.Text, nullable=True)
    patient_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        nullable=True
    )
    doctor_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        nullable=True
    )
    notes = db.Column(
        db.Text,
        nullable=True
    )
    patient = db.relationship(
        "User",
        foreign_keys=[patient_id],
        lazy="joined"
    )
    doctor = db.relationship(
        "User",
        foreign_keys=[doctor_id],
        lazy="joined"
    )
# ══════════════════════════════════════════════
# CREATE DATABASE TABLES
# ══════════════════════════════════════════════
with app.app_context():
    try:
        db.create_all()
        # create_all() does not add new columns to an existing SQLite table.
        # Apply the small schema additions used by the analysis-history flow
        # without requiring the user to delete their existing mdss.db.
        existing_columns = {
            row[1]
            for row in db.session.execute(
                text("PRAGMA table_info(analysis_result)")
            ).fetchall()
        }
        migration_columns = {
            "image_filename": "VARCHAR(200)",
            "subtlety": "INTEGER",
            "abnormality_type": "VARCHAR(50)",
            "image_view": "VARCHAR(20)",
            "breast_side": "VARCHAR(20)",
            "model_probability": "FLOAT",
            "shap_data": "TEXT"
        }
        for column_name, column_type in migration_columns.items():
            if column_name not in existing_columns:
                db.session.execute(
                    text(
                        f"ALTER TABLE analysis_result ADD COLUMN "
                        f"{column_name} {column_type}"
                    )
                )
        db.session.commit()
        print("Database tables created and schema checked successfully")
    except Exception as e:
        db.session.rollback()
        print(f"Database creation/migration error: {e}")
# ══════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════
def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )
def load_and_preprocess_image(image_path):
    """Load an image using the same VGG16 preprocessing used during training."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    model_input = preprocess_input(
        np.expand_dims(img_array, axis=0)
    )
    return model_input, img
def predict_with_model(
    img_array,
    clinical_features
):
    """Run the trained VGG16 + RBF-SVM hybrid model."""
    if hybrid_svm is None:
        raise RuntimeError(
            "The trained hybrid AI model is not loaded. "
            "Check the model folder and Python dependencies."
        )
    if vgg16_feature_extractor is None:
        raise RuntimeError(
            "The VGG16 feature extractor is not loaded."
        )
    # 1. Extract the 512 image features used during training.
    image_features = vgg16_feature_extractor.predict(
        img_array,
        verbose=0
    )
    # 2. Build the exact clinical feature tables used during training.
    #    DataFrames preserve the feature names expected by the fitted
    #    StandardScaler and OneHotEncoder, avoiding feature-name warnings.
    numeric_feature_names = list(
        getattr(
            clinical_scaler,
            "feature_names_in_",
            ["breast_density", "assessment", "subtlety"]
        )
    )
    numeric_features = pd.DataFrame(
        [[
            clinical_features["density"],
            clinical_features["assessment"],
            clinical_features["subtlety"]
        ]],
        columns=numeric_feature_names
    )
    categorical_feature_names = list(
        getattr(
            clinical_encoder,
            "feature_names_in_",
            [
                "abnormality_type",
                "image_view",
                "left_or_right_breast"
            ]
        )
    )
    categorical_features = pd.DataFrame(
        [[
            clinical_features["abnormality_type"],
            clinical_features["image_view"],
            clinical_features["breast_side"]
        ]],
        columns=categorical_feature_names
    )
    # 3. Apply the same preprocessing objects used during model training.
    numeric_scaled = clinical_scaler.transform(
        numeric_features
    )
    categorical_encoded = clinical_encoder.transform(
        categorical_features
    )
    if hasattr(categorical_encoded, "toarray"):
        categorical_encoded = categorical_encoded.toarray()
    # 5. Fuse VGG16 features + structured features.
    hybrid_features = np.concatenate(
        [
            image_features,
            numeric_scaled,
            categorical_encoded
        ],
        axis=1
    )
    expected_features = int(
        hybrid_svm.n_features_in_
    )
    if hybrid_features.shape[1] != expected_features:
        raise RuntimeError(
            "Hybrid feature dimension mismatch. "
            f"Expected {expected_features}, "
            f"got {hybrid_features.shape[1]}."
        )
    # 6. Apply the final SVM scaler used during training.
    hybrid_scaled = svm_scaler.transform(
        hybrid_features
    )
    # 7. Generate the final RBF-SVM prediction.
    prediction_numeric = int(
        hybrid_svm.predict(hybrid_scaled)[0]
    )
    probabilities = hybrid_svm.predict_proba(
        hybrid_scaled
    )[0]
    benign_probability = float(probabilities[0])
    malignant_probability = float(probabilities[1])
    prediction = (
        "MALIGNANT"
        if prediction_numeric == 1
        else "BENIGN"
    )
    confidence = round(
        max(
            benign_probability,
            malignant_probability
        ) * 100,
        1
    )
    return (
        prediction,
        confidence,
        malignant_probability,
        image_features
    )
def generate_gradcam_heatmap(
    image_path,
    prediction,
    output_filename
):
    """Generate a real Grad-CAM heatmap from VGG16's final convolutional layer.
    Note: because the final classifier is an RBF SVM, Grad-CAM explains the
    VGG16 image representation rather than the non-differentiable SVM decision
    itself.
    """
    if gradcam_model is None:
        raise RuntimeError(
            "Grad-CAM model is not loaded."
        )
    # Use the exact VGG16 preprocessing used by the feature extractor.
    img = Image.open(
        image_path
    ).convert("RGB")
    img_resized = img.resize((224, 224))
    img_array = np.array(
        img_resized,
        dtype=np.float32
    )
    model_input = preprocess_input(
        np.expand_dims(img_array, axis=0)
    )
    # Forward pass and gradient calculation.
    with tf.GradientTape() as tape:
        conv_outputs, vgg_outputs = gradcam_model(
            model_input,
            training=False
        )
        # VGG16 itself is being used as a feature extractor. We explain the
        # strength of the extracted representation at the final conv layer.
        target_output = tf.reduce_mean(
            vgg_outputs
        )
    gradients = tape.gradient(
        target_output,
        conv_outputs
    )
    if gradients is None:
        raise RuntimeError(
            "Grad-CAM could not calculate gradients."
        )
    conv_outputs = conv_outputs[0]
    gradients = gradients[0]
    # Global-average-pool the gradients to obtain channel weights.
    weights = tf.reduce_mean(
        gradients,
        axis=(0, 1)
    )
    heatmap = tf.reduce_sum(
        conv_outputs * weights,
        axis=-1
    )
    # ReLU and normalization.
    heatmap = tf.maximum(
        heatmap,
        0
    )
    heatmap_max = tf.reduce_max(
        heatmap
    )
    heatmap = tf.where(
        heatmap_max > 0,
        heatmap / heatmap_max,
        tf.zeros_like(heatmap)
    )
    heatmap = heatmap.numpy()
    # Resize the 14x14 VGG16 activation map to the mammogram size.
    heatmap_image = Image.fromarray(
        np.uint8(heatmap * 255)
    ).resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )
    heatmap = np.asarray(
        heatmap_image,
        dtype=np.float32
    ) / 255.0
    heatmap_colored = cm.jet(
        heatmap
    )[:, :, :3]
    heatmap_colored = (
        heatmap_colored * 255
    ).astype(np.uint8)
    alpha = 0.45
    overlay = (
        alpha * heatmap_colored
        + (1 - alpha) * np.asarray(
            img_resized,
            dtype=np.float32
        )
    ).clip(
        0,
        255
    ).astype(np.uint8)
    output_path = os.path.join(
        UPLOAD_FOLDER,
        output_filename
    )
    Image.fromarray(
        overlay
    ).save(output_path)
    return output_filename
def compute_shap_values(
    clinical_features,
    image_features
):
    """Compute real SHAP values for the nine structured model features.
    The current mammogram's 512 VGG16 image features are held fixed while
    SHAP KernelExplainer perturbs the nine structured features. Therefore the
    result explains the structured-feature contribution to the hybrid SVM
    malignant probability, while Grad-CAM separately explains the image
    representation.
    """
    if shap_background is None:
        raise RuntimeError(
            "Real SHAP background is not loaded. Add "
            "model/shap/shap_background.npy."
        )
    if clinical_scaler is None or svm_scaler is None or hybrid_svm is None:
        raise RuntimeError("Required trained model/scaler objects are not loaded.")
    current_raw = np.array([[
        clinical_features["density"],
        clinical_features["assessment"],
        clinical_features["subtlety"],
        1.0 if clinical_features["abnormality_type"] == "calcification" else 0.0,
        1.0 if clinical_features["abnormality_type"] == "mass" else 0.0,
        1.0 if clinical_features["image_view"] == "CC" else 0.0,
        1.0 if clinical_features["image_view"] == "MLO" else 0.0,
        1.0 if clinical_features["breast_side"] == "LEFT" else 0.0,
        1.0 if clinical_features["breast_side"] == "RIGHT" else 0.0
    ]], dtype=np.float32)
    background_raw = np.asarray(shap_background, dtype=np.float32)
    image_features_2d = np.asarray(image_features, dtype=np.float32).reshape(1, -1)
    if background_raw.shape[1] != 9:
        raise RuntimeError(
            f"SHAP background must contain 9 features; got {background_raw.shape[1]}."
        )
    if image_features_2d.shape[1] != 512:
        raise RuntimeError(
            f"Expected 512 VGG16 image features for SHAP; got {image_features_2d.shape[1]}."
        )
    def build_hybrid_features(structured_raw):
        structured_raw = np.asarray(structured_raw, dtype=np.float32)
        # Preserve the feature names used when the clinical scaler was fitted.
        numeric_feature_names = list(
            getattr(
                clinical_scaler,
                "feature_names_in_",
                ["breast_density", "assessment", "subtlety"]
            )
        )
        numeric_df = pd.DataFrame(
            structured_raw[:, :3],
            columns=numeric_feature_names
        )
        numeric_scaled = clinical_scaler.transform(numeric_df)
        return np.concatenate([
            np.repeat(image_features_2d, structured_raw.shape[0], axis=0),
            numeric_scaled,
            structured_raw[:, 3:]
        ], axis=1)
    def predict_malignant_probability(structured_raw):
        hybrid_features = build_hybrid_features(structured_raw)
        hybrid_scaled = svm_scaler.transform(hybrid_features)
        return hybrid_svm.predict_proba(hybrid_scaled)[:, 1]
    explainer = shap.KernelExplainer(
        predict_malignant_probability,
        background_raw
    )
    shap_result = explainer.shap_values(
        current_raw,
        nsamples=100
    )
    if isinstance(shap_result, list):
        shap_array = np.asarray(shap_result[-1])
    else:
        shap_array = np.asarray(shap_result)
    shap_array = np.squeeze(shap_array)
    if shap_array.ndim != 1 or shap_array.shape[0] != 9:
        raise RuntimeError(
            f"Unexpected SHAP output shape: {np.asarray(shap_result).shape}."
        )
    feature_names = [
        "Breast Density",
        "Assessment",
        "Subtlety",
        "Abnormality: Calcification",
        "Abnormality: Mass",
        "Image View: CC",
        "Image View: MLO",
        "Breast Side: Left",
        "Breast Side: Right"
    ]
    display_values = [
        clinical_features["density"],
        clinical_features["assessment"],
        clinical_features["subtlety"],
        clinical_features["abnormality_type"] == "calcification",
        clinical_features["abnormality_type"] == "mass",
        clinical_features["image_view"] == "CC",
        clinical_features["image_view"] == "MLO",
        clinical_features["breast_side"] == "LEFT",
        clinical_features["breast_side"] == "RIGHT"
    ]
    magnitudes = np.abs(shap_array)
    total = float(magnitudes.sum())
    percentages = (
        np.round(magnitudes / total * 100).astype(int)
        if total > 0 else np.zeros(9, dtype=int)
    )
    shap_features = []
    for name, input_value, contribution, percentage in zip(
        feature_names, display_values, shap_array, percentages
    ):
        shap_features.append({
            "name": name,
            "value": f"{float(contribution):+.4f}",
            "percentage": int(percentage),
            "direction": (
                "positive" if contribution > 0
                else "negative" if contribution < 0
                else "neutral"
            ),
            "input_value": str(input_value)
        })
    shap_features.sort(
        key=lambda item: abs(float(item["value"])),
        reverse=True
    )
    return shap_features
def compute_risk_level(
    risk_score,
    birads
):
    """
    Calculate the prototype MDSS risk index.
    This is NOT a calibrated clinical risk probability. It is a
    presentation-layer heuristic that combines the trained SVM's
    malignant probability with the entered BI-RADS category.
    """
    combined_risk = (
        risk_score * 0.7
        + (birads / 6) * 0.3
    )
    risk_percentage = round(
        combined_risk * 100,
        1
    )
    if combined_risk < 0.30:
        risk_level = "LOW"
        risk_color = "#2C6E49"
        recommendation = (
            "Review the mammogram and AI findings with a qualified "
            "healthcare professional. Further evaluation should be "
            "determined according to the patient's clinical context."
        )
    elif combined_risk < 0.70:
        risk_level = "INTERMEDIATE"
        risk_color = "#F5A623"
        recommendation = (
            "Review the mammogram, AI prediction, and clinical "
            "information with a qualified healthcare professional. "
            "Additional diagnostic evaluation may be considered "
            "according to clinical judgement."
        )
    else:
        risk_level = "HIGH"
        risk_color = "#C84B31"
        recommendation = (
            "Prompt review of the mammogram and AI findings by a "
            "qualified healthcare professional is warranted. "
            "Additional diagnostic evaluation should be determined "
            "by the treating clinical team."
        )
    return (
        risk_level,
        risk_color,
        recommendation,
        risk_percentage
    )
def radiologist_required(view_function):
    """Allow only recognized professional roles into clinical-management pages."""
    from functools import wraps

    @wraps(view_function)
    def wrapped(*args, **kwargs):
        if current_user.role == "patient":
            return redirect(url_for("patient_dashboard"))
        if current_user.role not in PROFESSIONAL_ROLES:
            abort(403)
        return view_function(*args, **kwargs)

    return wrapped
# ══════════════════════════════════════════════
# HOME ROUTE
# ══════════════════════════════════════════════
@app.route("/")
@login_required
@radiologist_required
def index():
    return render_template(
        "index.html",
        user=current_user
    )
# ══════════════════════════════════════════════
# PREDICTION ROUTE
# ══════════════════════════════════════════════
@app.route(
    "/predict",
    methods=["POST"]
)
@login_required
@radiologist_required
def predict():
    # ── Step 1: Get uploaded images ──
    if "images" not in request.files:
        return "No image uploaded.", 400
    files = request.files.getlist(
        "images"
    )
    files = [
        file
        for file in files
        if file.filename != ""
    ]
    if not files:
        return "No files selected.", 400
    timestamp_str = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    # ── Step 2: Save uploaded images ──
    image_paths = []
    for i, file in enumerate(files[:4]):
        if allowed_file(file.filename):
            filename = (
                f"mammogram_"
                f"{timestamp_str}_"
                f"{i}.jpg"
            )
            image_path = os.path.join(
                UPLOAD_FOLDER,
                filename
            )
            file.save(image_path)
            image_paths.append(
                image_path
            )
    if not image_paths:
        return "Invalid file type.", 400
    # ── Step 3: Use first image as primary image ──
    image_path = image_paths[0]
    # ── Step 4: Get clinical information ──
    try:
        clinical_features = {
            "age": int(
                request.form.get(
                    "age",
                    45
                )
            ),
            "birads": int(
                request.form.get(
                    "birads",
                    3
                )
            ),
            # The trained model uses BI-RADS as its assessment feature.
            "assessment": int(
                request.form.get(
                    "birads",
                    3
                )
            ),
            # These three fields are part of the actual CBIS-DDSM structured
            # feature vector used to train the SVM.
            "subtlety": int(
                request.form.get(
                    "subtlety",
                    3
                )
            ),
            "abnormality_type": request.form.get(
                "abnormality_type",
                "mass"
            ),
            "image_view": request.form.get(
                "image_view",
                "MLO"
            ),
            "breast_side": request.form.get(
                "breast_side",
                "LEFT"
            ),
            "family_history": int(
                request.form.get(
                    "family_history",
                    0
                )
            ),
            "density": int(
                request.form.get(
                    "density",
                    2
                )
            ),
            "menopause": int(
                request.form.get(
                    "menopause",
                    0
                )
            ),
            "prior_biopsy": int(
                request.form.get(
                    "prior_biopsy",
                    0
                )
            ),
            "notes": request.form.get(
                "notes",
                ""
            )
        }
    except ValueError:
        return (
            "Invalid clinical information provided.",
            400
        )
    # ── Step 5: Preprocess image ──
    # Correct indentation is important here.
    img_array, original_img = (
        load_and_preprocess_image(
            image_path
        )
    )
    # ── Step 6: Generate prediction ──
    try:
        (
            prediction,
            confidence,
            risk_score,
            image_features
        ) = predict_with_model(
            img_array,
            clinical_features
        )
    except Exception as e:
        return (
            f"AI model error: {str(e)}",
            500
        )
    # ── Step 7: Generate heatmaps ──
    heatmap_filenames = []
    for i, img_path in enumerate(image_paths):
        heatmap_filename = (
            f"heatmap_"
            f"{timestamp_str}_"
            f"{i}.jpg"
        )
        generate_gradcam_heatmap(
            img_path,
            prediction,
            heatmap_filename
        )
        heatmap_filenames.append(
            heatmap_filename
        )
    heatmap_filename = (
        heatmap_filenames[0]
    )
    # ── Step 8: Calculate SHAP-style values ──
    shap_features = compute_shap_values(
        clinical_features,
        image_features
    )
    # ── Step 9: Calculate risk level ──
    (
        risk_level,
        risk_color,
        recommendation,
        risk_percentage
    ) = compute_risk_level(
        risk_score,
        clinical_features["birads"]
    )
    # ── Step 10: Prepare display values ──
    prediction_class = (
        "malignant"
        if prediction == "MALIGNANT"
        else "benign"
    )
    family_history_display = (
        "Yes"
        if clinical_features["family_history"] == 1
        else "No"
    )
    menopause_display = (
        "Post-menopausal"
        if clinical_features["menopause"] == 1
        else "Pre-menopausal"
    )
    prior_biopsy_display = (
        "Yes"
        if clinical_features["prior_biopsy"] == 1
        else "No"
    )
    density_map = {
        1: "A (Almost entirely fatty)",
        2: "B (Scattered density)",
        3: "C (Heterogeneously dense)",
        4: "D (Extremely dense)"
    }
    density_display = density_map.get(
        clinical_features["density"],
        "B"
    )
    timestamp_display = datetime.now().strftime(
        "%d %B %Y, %H:%M"
    )
    # ── Step 11: Save result to database ──
    patient_id_raw = request.form.get("patient_id", "").strip()
    if not patient_id_raw:
        return (
            "Please select a registered patient before running the analysis.",
            400
        )
    try:
        patient_id = int(patient_id_raw)
    except ValueError:
        return "Invalid patient selected.", 400
    patient = db.session.get(User, patient_id)
    if patient is None or patient.role != "patient":
        return "Selected patient does not exist.", 400
    try:
        shap_json = json.dumps(shap_features)
        new_result = AnalysisResult(
            timestamp=timestamp_display,
            age=clinical_features["age"],
            birads=clinical_features["birads"],
            family_history=clinical_features["family_history"],
            density=clinical_features["density"],
            menopause=clinical_features["menopause"],
            prior_biopsy=clinical_features["prior_biopsy"],
            prediction=prediction,
            confidence=confidence,
            risk_level=risk_level,
            risk_score=risk_percentage,
            model_probability=round(risk_score * 100, 4),
            heatmap_filename=heatmap_filename,
            image_filename=os.path.basename(image_path),
            subtlety=clinical_features["subtlety"],
            abnormality_type=clinical_features["abnormality_type"],
            image_view=clinical_features["image_view"],
            breast_side=clinical_features["breast_side"],
            shap_data=shap_json,
            patient_id=patient_id,
            doctor_id=current_user.id,
            notes=clinical_features.get("notes", "")
        )
        db.session.add(new_result)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        # Avoid leaving orphaned generated files when the database write fails.
        for generated_path in image_paths + [
            os.path.join(UPLOAD_FOLDER, name)
            for name in heatmap_filenames
        ]:
            try:
                if os.path.exists(generated_path):
                    os.remove(generated_path)
            except OSError:
                pass
        print(f"Database save error: {e}")
        return (
            "The AI analysis completed, but the result could not be saved "
            "to the patient record. Please try again.",
            500
        )
    # ── Step 12: Display result page ──
    # result.html uses image_count to control the image/heatmap layout.
    # Pass the number of uploaded images explicitly so Jinja can render the page.
    image_count = len(image_paths)
    return render_template(
        "result.html",
        prediction=prediction,
        prediction_class=prediction_class,
        confidence=confidence,
        risk_score=risk_percentage,
        model_probability=round(
            risk_score * 100,
            1
        ),
        risk_percentage=risk_percentage,
        risk_level=risk_level,
        risk_color=risk_color,
        recommendation=recommendation,
        shap_features=shap_features,
        heatmap_filename=heatmap_filename,
        heatmap_filenames=heatmap_filenames,
        image_paths=image_paths,
        image_count=image_count,
        age=clinical_features["age"],
        birads=clinical_features["birads"],
        family_history=(
            family_history_display
        ),
        density=density_display,
        menopause=menopause_display,
        prior_biopsy=prior_biopsy_display,
        notes=clinical_features.get(
            "notes",
            ""
        ),
        timestamp=timestamp_display,
        analysis_id=new_result.id,
        patient_id=patient.id,
        patient_name=patient.name,
        user=current_user
    )
# ══════════════════════════════════════════════
# DASHBOARD ROUTE
# ══════════════════════════════════════════════
@app.route("/dashboard")
@login_required
@radiologist_required
def dashboard():
    try:
        all_results = AnalysisResult.query.filter_by(
            doctor_id=current_user.id
        ).order_by(
            AnalysisResult.id.desc()
        ).all()
        total = len(
            all_results
        )
        malignant = sum(
            1
            for result in all_results
            if result.prediction == "MALIGNANT"
        )
        benign = total - malignant
        avg_risk = (
            round(
                sum(
                    result.risk_score
                    for result in all_results
                ) / total,
                1
            )
            if total > 0
            else 0
        )
        high_risk = sum(
            1
            for result in all_results
            if result.risk_level == "HIGH"
        )
        intermediate_risk = sum(
            1
            for result in all_results
            if result.risk_level == "INTERMEDIATE"
        )
        low_risk = sum(
            1
            for result in all_results
            if result.risk_level == "LOW"
        )
        return render_template(
            "dashboard.html",
            all_results=all_results,
            total=total,
            malignant=malignant,
            benign=benign,
            avg_risk=avg_risk,
            high_risk=high_risk,
            intermediate_risk=intermediate_risk,
            low_risk=low_risk
        )
    except Exception as e:
        import traceback
        return (
            f"Dashboard error: "
            f"{traceback.format_exc()}",
            500
        )
# ══════════════════════════════════════════════
# DOWNLOAD PDF REPORT ROUTE
# ══════════════════════════════════════════════
@app.route("/download-report", methods=["POST"])
@login_required
def download_report():
    """Create a polished, multi-page MDSS PDF report."""
    prediction = (request.form.get("prediction") or "UNKNOWN").upper()[:30]
    confidence = (request.form.get("confidence") or "0")[:20]
    risk_level = (request.form.get("risk_level") or "UNKNOWN").upper()[:30]
    risk_score = (request.form.get("risk_score") or "0")[:20]
    recommendation = (request.form.get("recommendation") or
        "Review the mammogram and AI findings with a qualified healthcare professional.")[:4000]
    age = (request.form.get("age") or "Not provided")[:30]
    birads = (request.form.get("birads") or "Not provided")[:30]
    family_history = (request.form.get("family_history") or "Not provided")[:60]
    density = (request.form.get("density") or "Not provided")[:100]
    menopause = (request.form.get("menopause") or "Not provided")[:60]
    prior_biopsy = (request.form.get("prior_biopsy") or "Not provided")[:60]
    timestamp = (request.form.get("timestamp") or datetime.now().strftime("%d %B %Y, %H:%M"))[:100]
    patient_name = (request.form.get("patient_name") or "Not provided")[:120]
    patient_id = (request.form.get("patient_id") or "Not provided")[:40]
    analysis_id = (request.form.get("analysis_id") or "Not provided")[:40]
    heatmap_filename = os.path.basename(request.form.get("heatmap_filename") or "")
    shap_names = request.form.getlist("shap_names[]")[:20]
    shap_values = request.form.getlist("shap_values[]")[:20]

    # Palette: MDSS navy, blue and teal.
    NAVY = (8, 47, 91)
    BLUE = (8, 124, 193)
    TEAL = (22, 163, 181)
    PALE = (234, 245, 252)
    INK = (35, 54, 76)
    MUTED = (100, 116, 139)
    BORDER = (218, 230, 240)
    WHITE = (255, 255, 255)
    RED = (190, 55, 55)
    GREEN = (44, 110, 73)
    AMBER = (190, 125, 20)

    class MDSSReportPDF(FPDF):
        @staticmethod
        def _safe_pdf_text(value):
            """Convert text to characters supported by built-in Helvetica."""
            text_value = str(value if value is not None else "")
            replacements = {
                "\u2014": "-", "\u2013": "-", "\u2212": "-",
                "\u2018": "'", "\u2019": "'",
                "\u201c": '"', "\u201d": '"',
                "\u2022": "-", "\u00a0": " ",
                "\u2026": "...", "\u00d7": "x",
            }
            for old_char, new_char in replacements.items():
                text_value = text_value.replace(old_char, new_char)
            # Built-in FPDF core fonts use Latin-1. Replace any remaining
            # unsupported characters rather than raising FPDFUnicodeEncodingException.
            return text_value.encode("latin-1", errors="replace").decode("latin-1")

        def cell(self, w, h=0, text="", *args, **kwargs):
            return super().cell(w, h, self._safe_pdf_text(text), *args, **kwargs)

        def multi_cell(self, w, h, text, *args, **kwargs):
            return super().multi_cell(w, h, self._safe_pdf_text(text), *args, **kwargs)

        def rounded_rect(self, x, y, w, h, r=2, style=""):
            """FPDF-version-compatible rounded rectangle fallback.

            The installed PyFPDF/fpdf version may not implement rounded_rect.
            Draw a standard rectangle with the same fill/draw style so PDF
            generation remains compatible without requiring another package.
            """
            self.rect(x, y, w, h, style)

        def header(self):
            self.set_fill_color(*NAVY)
            self.rect(0, 0, 210, 27, "F")
            self.set_fill_color(*TEAL)
            self.rect(0, 27, 210, 1.5, "F")
            logo_path = os.path.join(basedir, "static", "images", "aaua-logo.png")
            if os.path.exists(logo_path):
                try:
                    self.image(logo_path, x=12, y=5, w=16, h=16)
                    title_x = 32
                except Exception:
                    title_x = 15
            else:
                title_x = 15
            self.set_text_color(*WHITE)
            self.set_font("Helvetica", "B", 14)
            self.set_xy(title_x, 5)
            self.cell(0, 8, "MDSS | BREAST CANCER ANALYSIS REPORT")
            self.set_font("Helvetica", "", 8)
            self.set_xy(title_x, 15)
            self.cell(0, 6, "Medical Decision Support System  |  Adekunle Ajasin University")
            self.set_y(35)

        def footer(self):
            self.set_y(-17)
            self.set_draw_color(*BORDER)
            self.line(15, self.get_y(), 195, self.get_y())
            self.set_y(-14)
            self.set_font("Helvetica", "", 8)
            self.set_text_color(*MUTED)
            self.cell(120, 7, "Confidential clinical support document | Review by a qualified professional")
            self.cell(60, 7, f"Page {self.page_no()}/{{nb}}", align="R")

    pdf = MDSSReportPDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_margins(15, 35, 15)
    pdf.set_auto_page_break(auto=True, margin=22)
    pdf.add_page()
    pdf.set_text_color(*INK)

    def section_heading(title):
        if pdf.get_y() > 255:
            pdf.add_page()
        y = pdf.get_y()
        pdf.set_fill_color(*PALE)
        pdf.set_draw_color(*BORDER)
        pdf.rounded_rect(15, y, 180, 9, 2, "DF")
        pdf.set_xy(19, y + 1)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*NAVY)
        pdf.cell(172, 7, title.upper())
        pdf.ln(13)

    def info_card(label, value, x, y, w=86):
        pdf.set_draw_color(*BORDER)
        pdf.set_fill_color(250, 252, 254)
        pdf.rounded_rect(x, y, w, 17, 2, "DF")
        pdf.set_xy(x + 4, y + 2)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*MUTED)
        pdf.cell(w - 8, 4, str(label)[:70])
        pdf.set_xy(x + 4, y + 7)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*INK)
        pdf.multi_cell(w - 8, 5, str(value)[:100])

    # Report metadata
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*MUTED)
    pdf.cell(100, 6, f"Generated: {timestamp}")
    pdf.cell(80, 6, f"Analysis reference: {analysis_id}", align="R")
    pdf.ln(10)

    # Summary panel
    pred_color = RED if prediction == "MALIGNANT" else GREEN if prediction == "BENIGN" else NAVY
    pdf.set_fill_color(*pred_color)
    pdf.rounded_rect(15, pdf.get_y(), 180, 25, 3, "F")
    pdf.set_xy(21, pdf.get_y() + 3)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(168, 8, f"AI PREDICTION: {prediction}")
    pdf.set_xy(21, pdf.get_y() + 1)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(168, 7, "Decision-support output — not a standalone diagnosis")
    pdf.ln(30)

    section_heading("Analysis overview")
    y = pdf.get_y()
    info_card("MODEL CONFIDENCE", f"{confidence}%", 15, y, 56)
    info_card("MDSS RISK INDEX", f"{risk_score}%", 77, y, 56)
    info_card("RISK CATEGORY", risk_level, 139, y, 56)
    pdf.set_y(y + 23)

    section_heading("Patient and clinical information")
    patient_rows = [
        ("Patient name", patient_name),
        ("Patient ID", patient_id),
        ("Age", f"{age} years" if age not in ("Not provided", "") else age),
        ("BI-RADS category", f"Category {birads}" if birads not in ("Not provided", "") else birads),
        ("Breast density", density),
        ("Family history", family_history),
        ("Menopausal status", menopause),
        ("Previous biopsy", prior_biopsy),
    ]
    for i in range(0, len(patient_rows), 2):
        y = pdf.get_y()
        left = patient_rows[i]
        right = patient_rows[i + 1] if i + 1 < len(patient_rows) else ("", "")
        info_card(left[0], left[1], 15, y, 86)
        if right[0]:
            info_card(right[0], right[1], 109, y, 86)
        pdf.set_y(y + 21)

    # Grad-CAM image with a short explanation.
    if heatmap_filename:
        heatmap_path = os.path.join(UPLOAD_FOLDER, heatmap_filename)
        if os.path.isfile(heatmap_path):
            section_heading("Grad-CAM visual explanation")
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(*MUTED)
            pdf.multi_cell(180, 5, "The overlay visualizes image regions highlighted by the model's feature representation. It should be interpreted alongside the original mammogram and clinical review.")
            pdf.ln(2)
            try:
                image_y = pdf.get_y()
                pdf.image(heatmap_path, x=55, y=image_y, w=100, h=70)
                pdf.set_y(image_y + 74)
            except Exception as heatmap_error:
                app.logger.warning("Could not embed Grad-CAM image in PDF: %s", heatmap_error)

    # SHAP table; values are passed from the result page's existing form.
    if shap_names:
        section_heading("SHAP feature contributions")
        pdf.set_fill_color(*NAVY)
        pdf.set_text_color(*WHITE)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(115, 8, "FEATURE", border=0, fill=True)
        pdf.cell(65, 8, "CONTRIBUTION VALUE", border=0, fill=True, ln=True, align="R")
        for index, (name, value) in enumerate(zip(shap_names, shap_values)):
            pdf.set_fill_color(*(PALE if index % 2 == 0 else WHITE))
            pdf.set_text_color(*INK)
            pdf.set_font("Helvetica", "", 8)
            pdf.cell(115, 7, str(name)[:100], border="B", fill=True)
            pdf.set_text_color(*TEAL)
            pdf.cell(65, 7, str(value)[:50], border="B", fill=True, ln=True, align="R")

    section_heading("Clinical review notes")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*INK)
    pdf.multi_cell(180, 5, recommendation or "No recommendation was supplied.")
    pdf.ln(5)

    # Medical disclaimer in a shaded panel.
    disclaimer = (
        "IMPORTANT DISCLAIMER: This AI-generated report is intended to support, not replace, "
        "professional clinical judgement. The MDSS Risk Index is a prototype heuristic and is not "
        "a calibrated clinical risk probability. The prediction and explanations must be reviewed "
        "by a qualified healthcare professional and must not be used as the sole basis for diagnosis "
        "or treatment decisions."
    )
    if pdf.get_y() > 245:
        pdf.add_page()
    y = pdf.get_y()
    pdf.set_fill_color(255, 248, 232)
    pdf.set_draw_color(238, 211, 157)
    pdf.rounded_rect(15, y, 180, 27, 2, "DF")
    pdf.set_xy(19, y + 3)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(130, 86, 15)
    pdf.cell(172, 5, "MEDICAL DISCLAIMER")
    pdf.set_xy(19, y + 9)
    pdf.set_font("Helvetica", "", 8)
    pdf.multi_cell(172, 4, disclaimer)

    pdf_output = io.BytesIO()
    pdf_bytes = pdf.output(dest="S")
    if isinstance(pdf_bytes, str):
        pdf_bytes = pdf_bytes.encode("latin-1")
    pdf_output.write(pdf_bytes)
    pdf_output.seek(0)
    return send_file(
        pdf_output,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="MDSS_Report_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".pdf",
    )

# ══════════════════════════════════════════════
# GOOGLE OAUTH ROUTES
# ══════════════════════════════════════════════
@app.route("/auth/google")
def google_login():
    if not GOOGLE_OAUTH_ENABLED:
        return (
            "Google login is not configured. Add GOOGLE_CLIENT_ID and "
            "GOOGLE_CLIENT_SECRET to your local .env file, then restart Flask.",
            503,
        )
    redirect_uri = url_for("google_callback", _external=True)
    return oauth.google.authorize_redirect(redirect_uri)
@app.route("/auth/google/callback")
def google_callback():
    if not GOOGLE_OAUTH_ENABLED:
        flash("Google login is not configured. Please use email and password.", "error")
        return redirect(url_for("login"))
    try:
        token = oauth.google.authorize_access_token()
        user_info = token.get("userinfo")
        if not user_info:
            user_info = oauth.google.userinfo()
        email = (user_info.get("email") or "").strip().lower()
        email_verified = user_info.get("email_verified", False)
        name = (user_info.get("name") or user_info.get("given_name") or "Google User").strip()
        if not email or email_verified is not True:
            flash("Google did not provide a verified email address. Please use email and password.", "error")
            return redirect(url_for("login"))
        user = User.query.filter_by(email=email).first()
        if user is None:
            # The existing User model requires a password, so a random,
            # unusable local password is stored for Google-only accounts.
            random_password = os.urandom(32).hex()
            password_hash = bcrypt.generate_password_hash(random_password).decode("utf-8")
            user = User(
                name=name[:100],
                email=email[:120],
                password=password_hash,
                role="patient",
                created_at=datetime.now().strftime("%d %B %Y"),
            )
            db.session.add(user)
            db.session.commit()
        login_user(user)
        if user.role == "patient":
            return redirect(url_for("patient_dashboard"))
        return redirect(url_for("index"))
    except Exception:
        app.logger.exception("Google OAuth sign-in failed")
        flash("Google sign-in failed. Please try again or sign in with your email and password.", "error")
        return redirect(url_for("login"))
# LOGIN ROUTE
# ══════════════════════════════════════════════
@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():
    if current_user.is_authenticated:
        if current_user.role == "patient":
            return redirect(
                "/patient-dashboard"
            )
        return redirect("/")
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        user = User.query.filter_by(
            email=email
        ).first()
        if user and bcrypt.check_password_hash(
            user.password,
            password
        ):
            login_user(
                user
            )
            if user.role == "patient":
                return redirect(
                    "/patient-dashboard"
                )
            return redirect("/")
        return render_template(
            "login.html",
            error=(
                "Invalid email or password. "
                "Please try again."
            )
        )
    return render_template(
        "login.html",
        error=None
    )
# ══════════════════════════════════════════════
# REGISTER ROUTE
# ══════════════════════════════════════════════
@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():
    if current_user.is_authenticated:
        if current_user.role == "patient":
            return redirect(url_for("patient_dashboard"))
        if current_user.role in PROFESSIONAL_ROLES:
            return redirect(url_for("index"))
        logout_user()

    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        confirm_password = request.form.get("confirm_password") or ""
        role = (request.form.get("role") or "patient").strip().lower()
        professional_key = request.form.get("professional_key") or ""

        def registration_error(message):
            return render_template("register.html", error=message), 400

        if not name or not email or not password:
            return registration_error("Please complete all required fields.")
        if len(name) > 100:
            return registration_error("Your name must be 100 characters or fewer.")
        if len(email) > 120:
            return registration_error("Your email address is too long.")
        if role not in VALID_ROLES:
            return registration_error("Please select a valid account role.")
        if len(password) < 8:
            return registration_error("Your password must contain at least 8 characters.")
        if password != confirm_password:
            return registration_error("The passwords do not match.")
        if request.form.get("agree_terms") != "on":
            return registration_error("Please confirm the responsible-use statement to continue.")

        # A professional role is never accepted without a valid server-side key.
        if role in PROFESSIONAL_ROLES:
            if not hmac.compare_digest(
                professional_key.strip(),
                PROFESSIONAL_ACCESS_KEY
            ):
                return registration_error(
                    "The Professional Access Key is invalid. Please check it and try again."
                )

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return registration_error(
                "An account with this email already exists. Please sign in instead."
            )

        hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")
        new_user = User(
            name=name,
            email=email,
            password=hashed_password,
            role=role,
            created_at=datetime.now().strftime("%d %B %Y")
        )

        try:
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user)
        except Exception:
            db.session.rollback()
            app.logger.exception("Account registration failed")
            return registration_error(
                "We could not create your account right now. Please try again."
            )

        if role == "patient":
            return redirect(url_for("patient_dashboard"))
        return redirect(url_for("index"))

    return render_template("register.html", error=None)

# ══════════════════════════════════════════════
# LOGOUT ROUTE
# ══════════════════════════════════════════════
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(
        "/login"
    )
# ══════════════════════════════════════════════
# PATIENT DASHBOARD ROUTE
# ══════════════════════════════════════════════
@app.route("/patient-dashboard")
@login_required
def patient_dashboard():
    if current_user.role != "patient":
        return redirect("/")
    results = AnalysisResult.query.filter_by(
        patient_id=current_user.id
    ).order_by(
        AnalysisResult.id.desc()
    ).all()
    return render_template(
        "patient_dashboard.html",
        user=current_user,
        results=results,
        total=len(results),
        latest=(
            results[0]
            if results
            else None
        ),
        high_risk=sum(
            1
            for result in results
            if result.risk_level == "HIGH"
        ),
        intermediate_risk=sum(
            1
            for result in results
            if result.risk_level == "INTERMEDIATE"
        ),
        low_risk=sum(
            1
            for result in results
            if result.risk_level == "LOW"
        )
    )
# ══════════════════════════════════════════════
# GET ALL PATIENTS
# ══════════════════════════════════════════════
@app.route("/get-patients")
@login_required
@radiologist_required
def get_patients():
    patients = User.query.filter_by(
        role="patient"
    ).all()
    return jsonify([
        {
            "id": patient.id,
            "name": patient.name,
            "email": patient.email
        }
        for patient in patients
    ])
# ══════════════════════════════════════════════
# ANALYSIS DETAIL ROUTE
# ══════════════════════════════════════════════
@app.route("/analysis/<int:analysis_id>")
@login_required
def analysis_detail(analysis_id):
    result = db.session.get(AnalysisResult, analysis_id)
    if result is None:
        abort(404)
    if current_user.role == "patient":
        if result.patient_id != current_user.id:
            return redirect(url_for("patient_dashboard"))
    elif current_user.role in PROFESSIONAL_ROLES:
        if result.doctor_id != current_user.id:
            # Professionals can open analyses they created.
            return redirect(url_for("dashboard"))
    else:
        abort(403)
    shap_features = []
    if result.shap_data:
        try:
            shap_features = json.loads(result.shap_data)
        except (TypeError, ValueError, json.JSONDecodeError):
            shap_features = []
    risk_level, risk_color, recommendation, risk_percentage = compute_risk_level(
        (result.model_probability or 0) / 100.0,
        result.birads
    )
    return render_template(
        "result.html",
        prediction=result.prediction,
        prediction_class=("malignant" if result.prediction == "MALIGNANT" else "benign"),
        confidence=result.confidence,
        risk_score=result.risk_score,
        model_probability=round(result.model_probability or 0, 1),
        risk_percentage=result.risk_score,
        risk_level=result.risk_level or risk_level,
        risk_color=risk_color,
        recommendation=recommendation,
        shap_features=shap_features,
        heatmap_filename=result.heatmap_filename,
        heatmap_filenames=[result.heatmap_filename] if result.heatmap_filename else [],
        image_paths=([url_for("static", filename=f"uploads/{result.image_filename}")] if result.image_filename else []),
        image_count=1 if result.image_filename else 0,
        age=result.age,
        birads=result.birads,
        family_history=("Yes" if result.family_history else "No"),
        density={1: "A (Almost entirely fatty)", 2: "B (Scattered density)", 3: "C (Heterogeneously dense)", 4: "D (Extremely dense)"}.get(result.density, "B"),
        menopause=("Post-menopausal" if result.menopause else "Pre-menopausal"),
        prior_biopsy=("Yes" if result.prior_biopsy else "No"),
        notes=result.notes or "",
        timestamp=result.timestamp,
        analysis_id=result.id,
        patient_id=result.patient_id,
        patient_name=result.patient.name if result.patient else "",
        user=current_user
    )
# ══════════════════════════════════════════════
# PATIENT HISTORY ROUTE
# ══════════════════════════════════════════════
@app.route(
    "/patient-history/<int:patient_id>"
)
@login_required
def patient_history(
    patient_id
):
    if current_user.role == "patient":
        if current_user.id != patient_id:
            return redirect(url_for("patient_dashboard"))
    elif current_user.role not in PROFESSIONAL_ROLES:
        abort(403)

    patient = db.session.get(User, patient_id)
    if patient is None:
        abort(404)
    results = AnalysisResult.query.filter_by(
        patient_id=patient_id
    ).order_by(
        AnalysisResult.id.desc()
    ).all()
    return render_template(
        "patient_history.html",
        patient=patient,
        results=results,
        total=len(results),
        high_risk=sum(
            1
            for result in results
            if result.risk_level == "HIGH"
        ),
        intermediate_risk=sum(
            1
            for result in results
            if result.risk_level == "INTERMEDIATE"
        ),
        low_risk=sum(
            1
            for result in results
            if result.risk_level == "LOW"
        ),
        latest=(
            results[0]
            if results
            else None
        )
    )

# ══════════════════════════════════════════════
# PATIENTS LIST PAGE
# ══════════════════════════════════════════════
@app.route("/patients")
@login_required
@radiologist_required
def patients_page():
    # Show patients associated with this professional's saved analyses, plus
    # registered patients not yet linked to an analysis so they can be selected.
    patients = User.query.filter_by(role="patient").order_by(User.name.asc()).all()
    patient_rows = []
    for patient in patients:
        count = AnalysisResult.query.filter_by(
            patient_id=patient.id,
            doctor_id=current_user.id
        ).count()
        latest = AnalysisResult.query.filter_by(
            patient_id=patient.id,
            doctor_id=current_user.id
        ).order_by(AnalysisResult.id.desc()).first()
        patient_rows.append({"patient": patient, "count": count, "latest": latest})
    return render_template("patients.html", patient_rows=patient_rows, user=current_user)

# ══════════════════════════════════════════════
# REPORTS LIST PAGE
# ══════════════════════════════════════════════
@app.route("/reports")
@login_required
@radiologist_required
def reports_page():
    reports = AnalysisResult.query.filter_by(
        doctor_id=current_user.id
    ).order_by(AnalysisResult.id.desc()).all()
    return render_template("reports.html", reports=reports, user=current_user)

# ══════════════════════════════════════════════
# SETTINGS PAGE
# ══════════════════════════════════════════════
@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings_page():
    if current_user.role not in VALID_ROLES:
        abort(403)
    if request.method == "POST":
        action = (request.form.get("action") or "profile").strip()
        if action == "profile":
            name = (request.form.get("name") or "").strip()
            email = (request.form.get("email") or "").strip().lower()
            if not name or not email:
                flash("Name and email are required.", "error")
            elif len(name) > 100 or len(email) > 120:
                flash("Name or email is too long.", "error")
            else:
                duplicate = User.query.filter(
                    User.email == email, User.id != current_user.id
                ).first()
                if duplicate:
                    flash("That email address is already in use.", "error")
                else:
                    current_user.name = name
                    current_user.email = email
                    db.session.commit()
                    flash("Your profile has been updated.", "success")
        elif action == "password":
            current_password = request.form.get("current_password") or ""
            new_password = request.form.get("new_password") or ""
            confirm_password = request.form.get("confirm_password") or ""
            if not bcrypt.check_password_hash(current_user.password, current_password):
                flash("Your current password is incorrect.", "error")
            elif len(new_password) < 8:
                flash("Your new password must contain at least 8 characters.", "error")
            elif new_password != confirm_password:
                flash("The new passwords do not match.", "error")
            else:
                current_user.password = bcrypt.generate_password_hash(new_password).decode("utf-8")
                db.session.commit()
                flash("Your password has been changed.", "success")
        else:
            abort(400)
        return redirect(url_for("settings_page"))
    return render_template("settings.html", user=current_user)

# ══════════════════════════════════════════════
# DASHBOARD SEARCH
# ══════════════════════════════════════════════
@app.route("/search")
@login_required
@radiologist_required
def dashboard_search():
    query = (request.args.get("q") or "").strip()[:100]
    results = []
    patients = []
    if query:
        term = f"%{query}%"
        results = AnalysisResult.query.filter(
            AnalysisResult.doctor_id == current_user.id,
            db.or_(
                AnalysisResult.prediction.ilike(term),
                AnalysisResult.risk_level.ilike(term),
                AnalysisResult.timestamp.ilike(term),
                AnalysisResult.notes.ilike(term)
            )
        ).order_by(AnalysisResult.id.desc()).limit(100).all()
        patients = User.query.filter(
            User.role == "patient",
            db.or_(User.name.ilike(term), User.email.ilike(term))
        ).order_by(User.name.asc()).limit(100).all()
    return render_template(
        "search_results.html", query=query, results=results,
        patients=patients, user=current_user
    )

# ══════════════════════════════════════════════
# BREASTCANCERMDSS LOCAL KNOWLEDGE-BASED ASSISTANT
# ══════════════════════════════════════════════
# This version runs without an external AI API or API credits.
# It does not query patient records, mammogram images, or model outputs.
# Replies are curated educational explanations selected by keyword matching.

CHATBOT_KNOWLEDGE = [
    {
        "topics": ("what is breast cancer", "define breast cancer", "breast cancer"),
        "answer": (
            "Breast cancer is a disease in which cells in breast tissue grow "
            "abnormally. A chatbot cannot determine whether someone has cancer. "
            "Assessment may involve clinical examination, imaging, and sometimes "
            "a biopsy. Discuss personal symptoms or results with a qualified "
            "healthcare professional."
        ),
    },
    {
        "topics": ("birads", "bi-rads", "breast imaging reporting"),
        "answer": (
            "BI-RADS means Breast Imaging Reporting and Data System. It is a "
            "standardized way for radiology reports to describe breast imaging "
            "findings and recommended follow-up. The category must be interpreted "
            "in context by a qualified clinician; it is not, by itself, a diagnosis."
        ),
    },
    {
        "topics": ("vgg16", "vgg 16"),
        "answer": (
            "VGG16 is a convolutional neural network used here as an image feature "
            "extractor. In BreastCancerMDSS, it converts the mammogram image into "
            "numerical features that are passed to the hybrid classifier. Its output "
            "does not independently confirm or exclude cancer."
        ),
    },
    {
        "topics": ("svm", "support vector machine", "rbf"),
        "answer": (
            "An SVM (Support Vector Machine) is a machine-learning algorithm that "
            "separates examples using patterns in their features. This project uses "
            "an RBF-kernel SVM with VGG16 image features and structured inputs. Its "
            "prediction is decision support only and requires clinical review."
        ),
    },
    {
        "topics": ("grad-cam", "grad cam", "heatmap", "heat map"),
        "answer": (
            "Grad-CAM creates a heatmap highlighting image regions associated with "
            "a neural network's feature representation. In this project, the heatmap "
            "helps visualize VGG16 image features; because the final classifier is "
            "an SVM, it should not be described as a direct explanation of the SVM "
            "decision. A heatmap is not proof that a region is cancerous."
        ),
    },
    {
        "topics": ("shap", "feature importance", "feature contribution"),
        "answer": (
            "SHAP is a method for explaining how input features contribute to a "
            "model output. In this project, the SHAP explanation focuses on the "
            "structured clinical/image metadata features while the image features "
            "are held fixed. Contributions describe model behavior, not causation "
            "or a confirmed clinical risk."
        ),
    },
    {
        "topics": ("confidence score", "confidence", "model probability", "probability"),
        "answer": (
            "The confidence value shown by BreastCancerMDSS summarizes the model's "
            "output for its predicted class. It is not the probability that a patient "
            "does or does not have cancer, and it should not be interpreted as a "
            "clinically calibrated measure unless validated for that purpose. A "
            "qualified professional must review the full case."
        ),
    },
    {
        "topics": ("risk score", "risk index", "risk level"),
        "answer": (
            "The MDSS Risk Index is a prototype presentation-layer score combining "
            "the model output with the entered BI-RADS category. It is a heuristic, "
            "not a calibrated clinical risk probability and not a diagnosis. Use it "
            "only alongside professional interpretation of the full clinical picture."
        ),
    },
    {
        "topics": ("benign", "malignant", "prediction", "result interpretation"),
        "answer": (
            "In general, 'benign' describes a non-cancerous finding and 'malignant' "
            "describes a cancerous finding. In BreastCancerMDSS these labels are "
            "model predictions, not confirmed diagnoses. Do not make treatment "
            "decisions from the prediction alone; a qualified healthcare professional "
            "must review the imaging and clinical context."
        ),
    },
    {
        "topics": ("patient data", "privacy", "private", "records", "data protection"),
        "answer": (
            "Please do not enter names, patient IDs, contact details, mammogram "
            "images, or other identifying medical information into this chat. This "
            "local assistant does not look up patient records or analysis results."
        ),
    },
    {
        "topics": ("new lump", "breast lump", "bloody nipple", "nipple discharge",
                   "skin changes", "symptom", "symptoms", "pain"),
        "answer": (
            "I cannot assess symptoms or determine their cause. A new breast lump, "
            "bloody nipple discharge, new skin or nipple changes, or other concerning "
            "breast symptoms should be assessed promptly by a qualified healthcare "
            "professional. If you believe you are experiencing a medical emergency, "
            "contact your local emergency service."
        ),
    },
    {
        "topics": ("treatment", "chemotherapy", "surgery", "medicine", "medication"),
        "answer": (
            "I cannot recommend a personal treatment plan or medication. Treatment "
            "depends on the confirmed diagnosis and individual clinical factors. Please "
            "discuss options with the treating healthcare team."
        ),
    },
    {
        "topics": ("how to use", "how does breastcancermdss work", "workflow", "system"),
        "answer": (
            "BreastCancerMDSS is a medical decision-support prototype. A healthcare "
            "professional selects a registered patient, provides the required "
            "structured information, and uploads a supported mammogram image. The "
            "VGG16 feature extractor and SVM produce a prediction; Grad-CAM and SHAP "
            "provide complementary explanations. All outputs require professional "
            "review and do not replace diagnosis."
        ),
    },
]

CHATBOT_FALLBACK = (
    "I can answer general questions about breast-health concepts and the "
    "BreastCancerMDSS workflow, including BI-RADS, VGG16, SVM, Grad-CAM, SHAP, "
    "confidence scores, and the prototype risk index. I cannot diagnose, interpret "
    "a personal case, or recommend treatment. Please avoid sharing identifying or "
    "sensitive medical information."
)

def local_chatbot_answer(message, user_role):
    """Return a curated local answer without contacting an external provider."""
    normalized = " ".join(message.lower().strip().split())
    # Prioritize urgent/symptom safety guidance before general topic matches.
    urgent_terms = (
        "new lump", "breast lump", "bloody nipple", "nipple discharge",
        "skin changes", "symptom", "symptoms"
    )
    if any(term in normalized for term in urgent_terms):
        return CHATBOT_KNOWLEDGE[10]["answer"]

    for item in CHATBOT_KNOWLEDGE:
        if any(topic in normalized for topic in item["topics"]):
            answer = item["answer"]
            if user_role in PROFESSIONAL_ROLES:
                answer += (
                    "\\n\\nProfessional note: interpret this educational "
                    "summary in context and follow your institution's clinical "
                    "protocols. The prototype is not a validated standalone "
                    "diagnostic device."
                )
            return answer
    return CHATBOT_FALLBACK

@app.route("/chatbot")
@login_required
def chatbot_page():
    if current_user.role not in VALID_ROLES:
        abort(403)
    return render_template("chatbot.html", user=current_user)

@app.route("/api/chatbot", methods=["POST"])
@login_required
def chatbot_api():
    """Authenticated local chatbot; no external API or patient-record lookup."""
    if current_user.role not in VALID_ROLES:
        abort(403)

    payload = request.get_json(silent=True) or {}
    incoming = payload.get("messages", [])
    if not isinstance(incoming, list):
        return jsonify({"error": "Invalid message format."}), 400

    # Accept only the latest user message; prior browser chat is not needed.
    latest_user_message = None
    for item in incoming[-8:]:
        if not isinstance(item, dict):
            continue
        if item.get("role") == "user" and isinstance(item.get("content"), str):
            content = item["content"].strip()
            if content and len(content) <= 1500:
                latest_user_message = content

    if not latest_user_message:
        return jsonify({"error": "Please enter a message to continue."}), 400

    try:
        answer = local_chatbot_answer(latest_user_message, current_user.role)
        return jsonify({"reply": answer, "mode": "local_knowledge"})
    except Exception:
        app.logger.exception("BreastCancerMDSS local chatbot failed")
        return jsonify({
            "error": "The assistant could not process that message. Please try again."
        }), 500

# ══════════════════════════════════════════════
# RUN APPLICATION
# ══════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 50)
    print(
        " MDSS - Breast Cancer Detection System"
    )
    print(
        " Subgroup 3 - AAUA CSC Dept"
    )
    print(
        " http://localhost:5000"
    )
    print("=" * 50)
    app.run(
        debug=True,
        port=5000
    )
