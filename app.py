# =====================================================
# APP.PY - PART 1
# IMPORTS + FLASK SETUP + MODEL LOADING
# =====================================================

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_file,
    redirect,
    url_for
)

import os
import json
import uuid
import joblib
import numpy as np
import pandas as pd

from datetime import datetime

# =====================================================
# PYTORCH
# =====================================================

import torch
import torch.nn as nn

from torchvision import (
    transforms,
    models
)

from PIL import Image

# =====================================================
# PDF GENERATION
# =====================================================

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table
)

from reportlab.lib.styles import (
    getSampleStyleSheet
)

# =====================================================
# FLASK APP
# =====================================================

app = Flask(__name__)

# =====================================================
# DIRECTORIES
# =====================================================

UPLOAD_FOLDER = "uploads"

REPORT_FOLDER = "reports"

PDF_FOLDER = "pdf_reports"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    REPORT_FOLDER,
    exist_ok=True
)

os.makedirs(
    PDF_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =====================================================
# LOAD CLINICAL MODEL
# =====================================================

print("Loading Clinical Model...")

xgb_model = joblib.load(
    "models/ckd_xgb_model.pkl"
)

label_encoders = joblib.load(
    "models/label_encoders.pkl"
)

feature_columns = joblib.load(
    "models/feature_columns.pkl"
)

print("Clinical Model Loaded")

# =====================================================
# LOAD CLASS NAMES
# =====================================================

class_names = joblib.load(
    "models/class_names.pkl"
)

print("Class Names Loaded")

# =====================================================
# DEVICE
# =====================================================

device = torch.device("cpu")

print(f"Device: {device}")

# =====================================================
# ULTRASOUND MODEL
# =====================================================

ultrasound_model = None
def get_ultrasound_model():

    global ultrasound_model

    if ultrasound_model is None:

        print("Loading Ultrasound Model...")

        ultrasound_model = models.resnet18(
            weights=None
        )

        num_features = (
            ultrasound_model.fc.in_features
        )

        ultrasound_model.fc = nn.Linear(
            num_features,
            2
        )

        ultrasound_model.load_state_dict(
            torch.load(
                "models/ultrasound_model.pth",
                map_location="cpu"
            )
        )

        ultrasound_model.eval()

        print("Ultrasound Model Loaded")

    return ultrasound_model

# =====================================================
# IMAGE TRANSFORM
# =====================================================

image_transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]

    )

])

print("Image Transform Loaded")

# =====================================================
# GLOBAL SETTINGS
# =====================================================

APP_NAME = (
    "KidneyAI"
)

PROJECT_TITLE = (
    "A Multimodal Chronic Kidney Disease "
    "Prediction Framework Using Clinical "
    "Parameters and Renal Ultrasound Imaging"
)

# =====================================================
# REPORT COUNTER
# =====================================================

def generate_report_id():

    return (
        "KID-" +
        str(uuid.uuid4())[:8].upper()
    )

# =====================================================
# STARTUP MESSAGE
# =====================================================

print("=" * 60)
print("KidneyAI Initialized Successfully")
print(PROJECT_TITLE)
print("=" * 60)

# =====================================================
# APP.PY - PART 2
# PREDICTION FUNCTIONS + REPORT FUNCTIONS
# =====================================================

# =====================================================
# CLINICAL MODEL PREDICTION
# =====================================================

def predict_clinical(patient_data):

    try:

        input_df = pd.DataFrame(
            [patient_data]
        )

        input_df = input_df.reindex(
            columns=feature_columns,
            fill_value=0
        )

        probability = (
            xgb_model.predict_proba(
                input_df
            )[0][1]
        )

        prediction = (
            xgb_model.predict(
                input_df
            )[0]
        )

        prediction_label = (
            "CKD"
            if prediction == 1
            else "Not CKD"
        )

        return {

            "prediction":
                prediction_label,

            "probability":
                float(
                    round(
                        float(probability) * 100,
                        2
                    )
                )

        }

    except Exception as e:

        print(
            f"Clinical Prediction Error: {e}"
        )

        return {

            "prediction":
                "Error",

            "probability":
                0

        }

# =====================================================
# ULTRASOUND PREDICTION
# =====================================================

def predict_ultrasound_image(
    image_path
):

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

        image_tensor = image_transform(
            image
        ).unsqueeze(0)

        image_tensor = image_tensor.to(
            device
        )

        with torch.no_grad():

            model = get_ultrasound_model()
            outputs = model(image)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence = (
                torch.max(
                    probabilities
                ).item()
            )

            predicted_class = torch.argmax(
                probabilities,
                dim=1
            ).item()

        prediction_label = (
            class_names[predicted_class]
        )

        return {

            "prediction":
                prediction_label,

            "probability":
                float(
                    round(
                        float(confidence) * 100,
                        2
                    )
                )

        }

    except Exception as e:

        print(
            f"Ultrasound Error: {e}"
        )

        return {

            "prediction":
                "Error",

            "probability":
                0

        }

# =====================================================
# MULTIMODAL FUSION
# =====================================================

def multimodal_fusion(

    clinical_score,
    ultrasound_score

):

    # =====================================================
    # IMPROVED MULTIMODAL FUSION STRATEGY
    # =====================================================
    
    # Convert percentages to probabilities (0-1)
    clinical_prob = clinical_score / 100.0
    ultrasound_prob = ultrasound_score / 100.0
    
    # Calculate adaptive weights based on confidence levels
    # Higher confidence models get higher weights
    clinical_weight = 0.3 + (abs(clinical_prob - 0.5) * 0.4)
    ultrasound_weight = 0.3 + (abs(ultrasound_prob - 0.5) * 0.4)
    
    # Normalize weights to sum to 1
    total_weight = clinical_weight + ultrasound_weight
    clinical_weight = clinical_weight / total_weight
    ultrasound_weight = ultrasound_weight / total_weight
    
    # Calculate consensus boost
    # If both models agree (both >60 or both <40), boost confidence
    agreement_score = 0.0
    if (clinical_prob > 0.60 and ultrasound_prob > 0.60):
        # Both agree on CKD
        agreement_score = 0.10
    elif (clinical_prob < 0.40 and ultrasound_prob < 0.40):
        # Both agree on Not CKD
        agreement_score = 0.10
    elif (0.40 <= clinical_prob <= 0.60 and 0.40 <= ultrasound_prob <= 0.60):
        # Both in uncertain zone
        agreement_score = -0.05
    else:
        # Disagreement
        agreement_score = -0.10
    
    # Weighted combination with harmonic mean for stability
    if clinical_prob > 0 and ultrasound_prob > 0:
        harmonic_mean = (
            2 * clinical_weight * clinical_prob * 
            ultrasound_weight * ultrasound_prob
        ) / (
            clinical_weight * clinical_prob + 
            ultrasound_weight * ultrasound_prob
        )
        # Blend weighted average with harmonic mean
        fusion_prob = (
            clinical_weight * clinical_prob + 
            ultrasound_weight * ultrasound_prob
        ) * 0.6 + harmonic_mean * 0.4
    else:
        fusion_prob = (
            clinical_weight * clinical_prob + 
            ultrasound_weight * ultrasound_prob
        )
    
    # Apply consensus boost (clamped between 0 and 1)
    fusion_prob = max(0.0, min(1.0, fusion_prob + agreement_score))
    
    # Convert back to percentage
    fusion_score = fusion_prob * 100.0
    
    # Decision threshold with confidence band
    if fusion_score >= 50:
        prediction = "CKD"
    else:
        prediction = "Not CKD"

    return {

        "prediction":
            prediction,

        "probability":
            float(
                round(
                    float(fusion_score),
                    2
                )
            )

    }

# =====================================================
# REPORT SAVE FUNCTION
# =====================================================

def save_report(

    prediction,
    probability,
    source,

    numerical_score=None,
    ultrasound_score=None,
    patient_data=None

):

    report_id = generate_report_id()

    # Determine risk level based on probability
    if probability >= 80:
        risk_level = "High Risk"
    elif probability >= 50:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Low Risk"

    report_data = {

        "report_id":
            report_id,

        "date":
            datetime.now().strftime(
                "%d-%m-%Y %H:%M:%S"
            ),

        "prediction":
            prediction,

        "probability":
            probability,

        "risk_level":
            risk_level,

        "source":
            source,

        "numerical_score":
            numerical_score,

        "ultrasound_score":
            ultrasound_score,

        "patient_data":
            patient_data if patient_data else {}

    }

    report_path = os.path.join(

        REPORT_FOLDER,

        f"{report_id}.json"

    )

    with open(

        report_path,

        "w"

    ) as file:

        json.dump(

            report_data,

            file,

            indent=4

        )

    return report_id

# =====================================================
# LOAD REPORT
# =====================================================

def load_report(
    report_id
):

    report_file = os.path.join(

        REPORT_FOLDER,

        f"{report_id}.json"

    )

    if not os.path.exists(
        report_file
    ):

        return None

    with open(
        report_file,
        "r"
    ) as file:

        return json.load(
            file
        )

# =====================================================
# LOAD ALL REPORTS
# =====================================================

def get_all_reports():

    reports = []

    for filename in os.listdir(
        REPORT_FOLDER
    ):

        if filename.endswith(
            ".json"
        ):

            file_path = os.path.join(

                REPORT_FOLDER,

                filename

            )

            try:

                with open(
                    file_path,
                    "r"
                ) as file:

                    reports.append(
                        json.load(
                            file
                        )
                    )

            except json.JSONDecodeError:

                print(
                    f"Warning: Corrupted report file: {filename}"
                )

                continue

    reports = sorted(

        reports,

        key=lambda x:
            x["date"],

        reverse=True

    )

    return reports

# =====================================================
# PDF GENERATOR
# =====================================================

def generate_pdf_report(

    report_id

):

    report = load_report(
        report_id
    )

    if report is None:

        return None

    pdf_path = os.path.join(

        PDF_FOLDER,

        f"{report_id}.pdf"

    )

    doc = SimpleDocTemplate(
        pdf_path,
        topMargin=20,
        bottomMargin=20,
        leftMargin=20,
        rightMargin=20
    )

    styles = getSampleStyleSheet()

    # Create custom styles
    title_style = styles["Heading1"]
    title_style.fontSize = 24
    title_style.textColor = styles["Title"].textColor
    title_style.spaceAfter = 6

    subtitle_style = styles["Heading2"]
    subtitle_style.fontSize = 10
    subtitle_style.textColor = styles["Normal"].textColor
    subtitle_style.spaceAfter = 12

    section_style = styles["Heading2"]
    section_style.fontSize = 12
    section_style.textColor = styles["Title"].textColor
    section_style.spaceAfter = 10

    content = []

    # REPORT HEADER
    content.append(

        Paragraph(

            "KidneyAI Diagnostic Report",

            title_style

        )

    )

    content.append(

        Paragraph(

            "AI-Powered Chronic Kidney Disease Assessment",

            subtitle_style

        )

    )

    content.append(
        Spacer(1, 12)
    )

    # REPORT INFORMATION
    content.append(

        Paragraph(

            "Report Information",

            section_style

        )

    )

    info_text = (

        f"<b>Report ID:</b> {report['report_id']}<br/>"
        f"<b>Date & Time:</b> {report['date']}<br/>"
        f"<b>Assessment Source:</b> {report['source']}<br/>"
        f"<b>Risk Level:</b> {report.get('risk_level', 'N/A')}"

    )

    content.append(

        Paragraph(
            info_text,
            styles["BodyText"]
        )

    )

    content.append(
        Spacer(1, 12)
    )

    # ANALYSIS SUMMARY
    content.append(

        Paragraph(

            "Analysis Summary",

            section_style

        )

    )

    summary_text = (

        f"<b>Prediction Status:</b> {report['prediction']}<br/>"
        f"<b>Confidence Score:</b> {report['probability']}%<br/>"

    )

    if report.get('numerical_score') is not None:

        summary_text += (

            f"<b>Clinical Score:</b> "
            f"{report['numerical_score']}%<br/>"

        )

    if report.get('ultrasound_score') is not None:

        summary_text += (

            f"<b>Ultrasound Score:</b> "
            f"{report['ultrasound_score']}%"

        )

    content.append(

        Paragraph(
            summary_text,
            styles["BodyText"]
        )

    )

    content.append(
        Spacer(1, 12)
    )

    # DIAGNOSIS BOX
    diagnosis_title = (

        "Chronic Kidney Disease Suspected"

        if report['prediction'] == "CKD"

        else "No Evidence of CKD"

    )

    diagnosis_text = (

        f"<b>{diagnosis_title}</b><br/>"
        f"Confidence: {report['probability']}%"

    )

    content.append(

        Paragraph(
            diagnosis_text,
            styles["BodyText"]
        )

    )

    content.append(
        Spacer(1, 12)
    )

    # PATIENT DATA TABLE
    if report.get('patient_data'):

        content.append(

            Paragraph(

                "Patient Clinical Parameters",

                section_style

            )

        )

        param_display = {

            'age': 'Age',
            'bp': 'Blood Pressure',
            'sg': 'Specific Gravity',
            'al': 'Albumin',
            'su': 'Sugar',
            'bgr': 'Blood Glucose Random',
            'bu': 'Blood Urea',
            'sc': 'Serum Creatinine',
            'sod': 'Sodium',
            'pot': 'Potassium',
            'hemo': 'Hemoglobin',
            'pcv': 'Packed Cell Volume',
            'wc': 'White Blood Cell Count',
            'rc': 'Red Blood Cell Count',
            'htn': 'Hypertension',
            'dm': 'Diabetes Mellitus',
            'cad': 'Coronary Artery Disease',
            'pe': 'Pedal Edema',
            'ane': 'Anemia'

        }

        table_data = [

            ['Parameter', 'Value']

        ]

        for param, value in (

            report['patient_data'].items()

        ):

            if param in param_display:

                table_data.append([

                    param_display[param],

                    str(value)

                ])

        patient_table = Table(table_data)

        content.append(patient_table)

        content.append(
            Spacer(1, 12)
        )

    # CLINICAL RECOMMENDATIONS
    content.append(

        Paragraph(

            "Clinical Recommendations",

            section_style

        )

    )

    recommendations = []

    if (report['prediction'] == "CKD" 

        and report['probability'] >= 80):

        recommendations = [

            "Schedule urgent consultation with nephrologist",

            "Perform additional diagnostic tests",

            "Monitor kidney function regularly",

            "Review medication adjustments"

        ]

    elif (report['prediction'] == "CKD" 

        and report['probability'] >= 50):

        recommendations = [

            "Schedule routine consultation with nephrologist",

            "Perform follow-up diagnostic tests",

            "Monitor clinical parameters regularly",

            "Implement lifestyle modifications"

        ]

    else:

        recommendations = [

            "Continue regular health monitoring",

            "Maintain healthy lifestyle habits",

            "Schedule periodic clinical assessments",

            "Follow preventive health guidelines"

        ]

    rec_text = "<br/>".join([

        f"• {rec}" for rec in recommendations

    ])

    content.append(

        Paragraph(
            rec_text,
            styles["BodyText"]
        )

    )

    content.append(
        Spacer(1, 12)
    )

    # CLINICAL INTERPRETATION
    content.append(

        Paragraph(

            "Clinical Interpretation",

            section_style

        )

    )

    interpretation = (

        "This AI-powered diagnostic report provides "
        "clinical decision support based on the analysis of "
        "patient clinical parameters and/or renal ultrasound imaging. "
        "The assessment is designed to assist healthcare professionals "
        "in identifying potential cases of Chronic Kidney Disease. "
        "However, this report should be interpreted in conjunction with "
        "comprehensive clinical evaluation, laboratory results, and imaging studies. "
        "A qualified healthcare professional must review all findings and clinical "
        "context before making any diagnostic or treatment decisions."

    )

    content.append(

        Paragraph(
            interpretation,
            styles["BodyText"]
        )

    )

    content.append(
        Spacer(1, 12)
    )

    # FOOTER
    footer_text = (

        f"<i>Generated by KidneyAI | "
        f"Multimodal Chronic Kidney Disease Prediction Framework | "
        f"{report['report_id']}</i>"

    )

    content.append(

        Paragraph(
            footer_text,
            styles["Normal"]
        )

    )

    doc.build(
        content
    )

    return pdf_path

# =====================================================
# DASHBOARD STATS
# =====================================================

def get_dashboard_stats():

    reports = get_all_reports()

    total_reports = len(
        reports
    )

    total_ckd = len([

        r for r in reports

        if str(
            r["prediction"]
        ).lower() == "ckd"

    ])

    total_normal = len([

        r for r in reports

        if str(
            r["prediction"]
        ).lower() in [
            "not ckd",
            "normal"
        ]

    ])

    return {

        "total_reports":
            total_reports,

        "total_ckd":
            total_ckd,

        "total_normal":
            total_normal

    }

# =====================================================
# END OF PART 2
# =====================================================

# =====================================================
# APP.PY - PART 3
# ROUTES
# =====================================================

# =====================================================
# DASHBOARD
# =====================================================
@app.route("/")
def dashboard():
    return render_template("dashboard_v2.html")

@app.route("/numerical")
def numerical():
    return render_template("numerical_v2.html")

@app.route("/ultrasound")
def ultrasound():
    return render_template("ultrasound_v2.html")

@app.route("/combined")
def combined():
    return render_template("combined_v2.html")

@app.route("/reports")
def reports():
    all_reports = get_all_reports()
    stats = get_dashboard_stats()
    return render_template(
        "reports_v2.html",
        reports=all_reports,
        stats=stats
    )


# =====================================================
# PAGE ROUTES
# =====================================================



# =====================================================
# ROUTE ALIASES (_v2 sidebar links)
# =====================================================

@app.route("/dashboard_v2")
def dashboard_v2():
    return render_template("dashboard_v2.html")

@app.route("/numerical_v2")
def numerical_v2():
    return render_template("numerical_v2.html")

@app.route("/ultrasound_v2")
def ultrasound_v2():
    return render_template("ultrasound_v2.html")

@app.route("/combined_v2")
def combined_v2():
    return render_template("combined_v2.html")

@app.route("/reports_v2")
def reports_v2():
    all_reports = get_all_reports()
    stats = get_dashboard_stats()
    return render_template(
        "reports_v2.html",
        reports=all_reports,
        stats=stats
    )

@app.route("/result_v2")
def result_v2():

    return render_template(

        "result_v2.html",

        report_id="KID-001",

        date=datetime.now().strftime("%d-%m-%Y"),

        age=45,

        clinical_prediction="CKD Detected",

        clinical_confidence="96%",

        ultrasound_prediction="Abnormal",

        ultrasound_confidence="94%",

        fusion_prediction="High Risk CKD",

        fusion_confidence="98%",

        risk_level="HIGH"

    )


# =====================================================
# CLINICAL PREDICTION
# =====================================================

@app.route(
    "/predict_numerical",
    methods=["POST"]
)
def predict_numerical():

    try:

        patient_data = {}

        for column in feature_columns:

            value = request.form.get(
                column,
                0
            )

            if value == "":
                value = 0

            patient_data[column] = float(
                value
            )

        result = predict_clinical(
            patient_data
        )

        report_id = save_report(

            prediction=result[
                "prediction"
            ],

            probability=result[
                "probability"
            ],

            source="Clinical Model",

            numerical_score=result[
                "probability"
            ],

            patient_data=patient_data

        )

        from datetime import datetime

        return render_template(

            "result_v2.html",

            report_id=report_id,

            date=datetime.now().strftime("%d-%m-%Y"),

            age=patient_data.get("age", "N/A"),

            clinical_prediction=result["prediction"],

            clinical_confidence=f"{result['probability']:.2f}%",

            ultrasound_prediction="Not Available",

            ultrasound_confidence="--",

            fusion_prediction=result["prediction"],

            fusion_confidence=f"{result['probability']:.2f}%",

            risk_level="HIGH" if result["probability"] > 70 else "LOW",

            patient_data=patient_data

        )

    except Exception as e:

        return f"Clinical Prediction Error: {e}"

# =====================================================
# ULTRASOUND PREDICTION
# =====================================================

@app.route(
    "/predict_ultrasound",
    methods=["POST"]
)
def predict_ultrasound():

    try:

        if "image" not in request.files:

            return "No Image Uploaded"

        image = request.files[
            "image"
        ]

        filename = (

            str(uuid.uuid4())

            + "_"

            + image.filename

        )

        image_path = os.path.join(

            UPLOAD_FOLDER,

            filename

        )

        image.save(
            image_path
        )

        result = predict_ultrasound_image(
            image_path
        )

        report_id = save_report(

            prediction=result[
                "prediction"
            ],

            probability=result[
                "probability"
            ],

            source="Ultrasound CNN",

            ultrasound_score=result[
                "probability"
            ]

        )

        from datetime import datetime

        return render_template(

            "result_v2.html",

            report_id=report_id,

            date=datetime.now().strftime("%d-%m-%Y"),

            age="N/A",

            clinical_prediction="Not Available",

            clinical_confidence="--",

            ultrasound_prediction=result["prediction"],

            ultrasound_confidence=f"{result['probability']:.2f}%",

            fusion_prediction=result["prediction"],

            fusion_confidence=f"{result['probability']:.2f}%",

            risk_level="MEDIUM"

        )

    except Exception as e:

        return f"Ultrasound Error: {e}"

# =====================================================
# MULTIMODAL PREDICTION
# =====================================================

@app.route(
    "/predict_combined",
    methods=["POST"]
)
def predict_combined():

    try:

        # ======================
        # CLINICAL INPUT
        # ======================

        patient_data = {}

        for column in feature_columns:

            value = request.form.get(
                column,
                0
            )

            if value == "":
                value = 0

            patient_data[column] = float(
                value
            )

        clinical_result = (

            predict_clinical(
                patient_data
            )

        )

        # ======================
        # IMAGE INPUT
        # ======================

        image = request.files[
            "image"
        ]

        filename = (

            str(uuid.uuid4())

            + "_"

            + image.filename

        )

        image_path = os.path.join(

            UPLOAD_FOLDER,

            filename

        )

        image.save(
            image_path
        )

        ultrasound_result = (

            predict_ultrasound_image(
                image_path
            )

        )

        # ======================
        # FUSION
        # ======================

        fusion_result = (

            multimodal_fusion(

                clinical_result[
                    "probability"
                ],

                ultrasound_result[
                    "probability"
                ]

            )

        )

        report_id = save_report(

            prediction=fusion_result[
                "prediction"
            ],

            probability=fusion_result[
                "probability"
            ],

            source="Multimodal Fusion",

            numerical_score=clinical_result[
                "probability"
            ],

            ultrasound_score=ultrasound_result[
                "probability"
            ],

            patient_data=patient_data

        )

        from datetime import datetime

        risk_level = "LOW"

        if fusion_result["probability"] >= 70:
            risk_level = "HIGH"
        elif fusion_result["probability"] >= 40:
            risk_level = "MEDIUM"

        return render_template(

            "result_v2.html",

            report_id=report_id,

            date=datetime.now().strftime("%d-%m-%Y"),

            age=patient_data.get("age", "N/A"),

            clinical_prediction=clinical_result["prediction"],

            clinical_confidence=f"{clinical_result['probability']:.2f}%",

            ultrasound_prediction=ultrasound_result["prediction"],

            ultrasound_confidence=f"{ultrasound_result['probability']:.2f}%",

            fusion_prediction=fusion_result["prediction"],

            fusion_confidence=f"{fusion_result['probability']:.2f}%",

            risk_level=risk_level,

            patient_data=patient_data

        )

    except Exception as e:

        return f"Fusion Error: {e}"

# =====================================================
# VIEW REPORT
# =====================================================

@app.route(
    "/report/<report_id>"
)
def view_report(
    report_id
):

    report = load_report(
        report_id
    )

    if report is None:

        return "Report Not Found"

    clinical_score = report.get("numerical_score")
    ultrasound_score = report.get("ultrasound_score")

    return render_template(
        "result_v2.html",
        report_id=report["report_id"],
        date=report["date"],
        age=report.get("patient_data", {}).get("age", "N/A"),
        clinical_prediction=(
            report["prediction"] if clinical_score is not None else "Not Available"
        ),
        clinical_confidence=(
            f"{clinical_score:.2f}%" if clinical_score is not None else "--"
        ),
        ultrasound_prediction=(
            report["prediction"] if ultrasound_score is not None else "Not Available"
        ),
        ultrasound_confidence=(
            f"{ultrasound_score:.2f}%" if ultrasound_score is not None else "--"
        ),
        fusion_prediction=report["prediction"],
        fusion_confidence=f"{report['probability']:.2f}%",
        risk_level=report.get("risk_level", "N/A").upper().replace(" RISK", ""),
        patient_data=report.get("patient_data", {})
    )

# =====================================================
# DOWNLOAD PDF
# =====================================================

@app.route(
    "/download_report/<report_id>"
)
def download_report(
    report_id
):

    pdf_path = generate_pdf_report(
        report_id
    )

    if pdf_path is None:

        return "PDF Not Found"

    return send_file(

        pdf_path,

        as_attachment=True

    )

# =====================================================
# API STATUS
# =====================================================

@app.route("/health")
def health():

    return jsonify({

        "status":
            "running",

        "application":
            APP_NAME

    })

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    print(
        "\nStarting KidneyAI Server..."
    )

    app.run(
    host="0.0.0.0",
    port=5000,
    debug=True,
    use_reloader=False
)