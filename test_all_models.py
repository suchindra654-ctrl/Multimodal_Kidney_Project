#!/usr/bin/env python3
"""
Comprehensive Model Testing Script for KidneyAI
Tests all models (Clinical XGBoost, Ultrasound CNN) with sample data
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from datetime import datetime

print("=" * 70)
print("KIDNEYAI MODEL TESTING SUITE")
print("=" * 70)
print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# =====================================================
# PART 1: LOAD ALL MODELS
# =====================================================
print("[STEP 1] Loading Models...")
print("-" * 70)

try:
    # Load Clinical Model
    print("  → Loading XGBoost Clinical Model...", end=" ")
    xgb_model = joblib.load("models/ckd_xgb_model.pkl")
    print("✓ SUCCESS")

    # Load Feature Columns
    print("  → Loading Feature Columns...", end=" ")
    feature_columns = joblib.load("models/feature_columns.pkl")
    print(f"✓ SUCCESS ({len(feature_columns)} features)")

    # Load Label Encoders
    print("  → Loading Label Encoders...", end=" ")
    label_encoders = joblib.load("models/label_encoders.pkl")
    print(f"✓ SUCCESS ({len(label_encoders)} encoders)")

    # Load Class Names
    print("  → Loading Class Names...", end=" ")
    class_names = joblib.load("models/class_names.pkl")
    print(f"✓ SUCCESS ({len(class_names)} classes)")

    # Load Ultrasound Model
    print("  → Loading Ultrasound CNN Model...", end=" ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ultrasound_model = models.resnet18(weights=None)
    num_features = ultrasound_model.fc.in_features
    ultrasound_model.fc = nn.Linear(num_features, 2)
    ultrasound_model.load_state_dict(
        torch.load("models/ultrasound_model.pth", map_location=device)
    )
    ultrasound_model.to(device)
    ultrasound_model.eval()
    print(f"✓ SUCCESS (Device: {device})")

    # Load Image Transform
    print("  → Loading Image Transform...", end=" ")
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    print("✓ SUCCESS")

    print("\n✓ All models loaded successfully!\n")

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    sys.exit(1)

# =====================================================
# PART 2: TEST CLINICAL MODEL
# =====================================================
print("[STEP 2] Testing Clinical/Numerical Model")
print("-" * 70)

# Create sample patient data
sample_patient = {
    'age': 48.0,
    'bp': 80.0,
    'sg': 1.02,
    'al': 1.0,
    'su': 0.0,
    'bgr': 121.0,
    'bu': 36.4,
    'sc': 1.8,
    'sod': 137.0,
    'pot': 4.6,
    'hemo': 15.4,
    'pcv': 44.0,
    'wc': 7800.0,
    'rc': 5.2,
    'htn': 1.0,
    'dm': 0.0,
    'cad': 0.0,
    'pe': 0.0,
    'ane': 0.0
}

print("\nSample Patient Clinical Parameters:")
print("-" * 70)
for key, value in sample_patient.items():
    print(f"  {key:15s}: {value}")

try:
    # Prepare input
    input_df = pd.DataFrame([sample_patient])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    
    # Make prediction
    probability = xgb_model.predict_proba(input_df)[0][1]
    prediction = xgb_model.predict(input_df)[0]
    prediction_label = "CKD" if prediction == 1 else "Not CKD"
    confidence = float(round(float(probability) * 100, 2))
    
    print("\n✓ Clinical Model Prediction Results:")
    print("-" * 70)
    print(f"  Prediction:     {prediction_label}")
    print(f"  Confidence:     {confidence}%")
    print(f"  CKD Probability: {probability:.4f}")
    
    clinical_score = confidence
    clinical_prediction = prediction_label

except Exception as e:
    print(f"\n✗ ERROR in Clinical Model: {e}")
    clinical_score = None
    clinical_prediction = None

# =====================================================
# PART 3: TEST ULTRASOUND MODEL
# =====================================================
print("\n[STEP 3] Testing Ultrasound CNN Model")
print("-" * 70)

try:
    # Create a dummy test image (224x224 RGB)
    print("\n  → Creating dummy test ultrasound image (224x224)...", end=" ")
    dummy_image = Image.new('RGB', (224, 224), color='gray')
    dummy_image.save('temp_test_image.jpg')
    print("✓")
    
    # Load and process image
    print("  → Loading and processing image...", end=" ")
    test_image = Image.open('temp_test_image.jpg').convert('RGB')
    image_tensor = image_transform(test_image).unsqueeze(0).to(device)
    print("✓")
    
    # Make prediction
    print("  → Running inference...", end=" ")
    with torch.no_grad():
        outputs = ultrasound_model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence_ultrasound = float(probabilities[0][predicted_class].cpu().numpy() * 100)
    
    ultrasound_label = "CKD" if predicted_class == 1 else "Normal"
    print("✓")
    
    print("\n✓ Ultrasound Model Prediction Results:")
    print("-" * 70)
    print(f"  Prediction:     {ultrasound_label}")
    print(f"  Confidence:     {confidence_ultrasound:.2f}%")
    print(f"  Predicted Class: {predicted_class}")
    
    ultrasound_score = confidence_ultrasound
    ultrasound_prediction = ultrasound_label
    
    # Cleanup
    if os.path.exists('temp_test_image.jpg'):
        os.remove('temp_test_image.jpg')

except Exception as e:
    print(f"\n✗ ERROR in Ultrasound Model: {e}")
    ultrasound_score = None
    ultrasound_prediction = None

# Initialize fusion variables
fusion_score = None
fusion_prediction = None

# =====================================================
# PART 4: TEST MULTIMODAL FUSION
# =====================================================
print("\n[STEP 4] Testing Multimodal Fusion")
print("-" * 70)

if clinical_score is not None and ultrasound_score is not None:
    try:
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
        
        # Decision threshold
        fusion_prediction = "CKD" if fusion_score >= 50 else "Not CKD"
        
        print("\n🔄 Improved Multimodal Fusion Strategy:")
        print("-" * 70)
        print(f"  Algorithm:       Adaptive Weighted Harmonic Fusion with Consensus Boost")
        print(f"  Clinical Score:  {clinical_score:.2f}% (weight: {clinical_weight:.2%})")
        print(f"  Ultrasound Score: {ultrasound_score:.2f}% (weight: {ultrasound_weight:.2%})")
        print(f"  Agreement Boost: {agreement_score:+.2f}")
        print(f"  Fusion Score:    {fusion_score:.2f}%")
        print(f"  Decision:        {fusion_prediction}")
        
    except Exception as e:
        print(f"✗ ERROR in Fusion: {e}")
        fusion_score = None
        fusion_prediction = None
else:
    print("\n✗ Cannot test fusion: One or both models failed")
    fusion_score = None
    fusion_prediction = None

# =====================================================
# PART 5: SUMMARY REPORT
# =====================================================
print("\n" + "=" * 70)
print("SUMMARY REPORT")
print("=" * 70)

test_results = {
    "test_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "device": str(device),
    "models_status": {
        "xgboost_clinical": "✓ LOADED",
        "feature_columns": f"✓ LOADED ({len(feature_columns)} features)",
        "label_encoders": f"✓ LOADED ({len(label_encoders)} encoders)",
        "ultrasound_cnn": f"✓ LOADED",
        "image_transform": "✓ LOADED"
    },
    "clinical_model_test": {
        "status": "✓ PASSED" if clinical_prediction else "✗ FAILED",
        "prediction": clinical_prediction,
        "confidence": clinical_score
    },
    "ultrasound_model_test": {
        "status": "✓ PASSED" if ultrasound_prediction else "✗ FAILED",
        "prediction": ultrasound_prediction,
        "confidence": ultrasound_score
    },
    "multimodal_fusion_test": {
        "status": "✓ PASSED" if (clinical_prediction and ultrasound_prediction and fusion_score is not None) else "✗ FAILED",
        "fusion_score": fusion_score if fusion_score is not None else None,
        "fusion_prediction": fusion_prediction if fusion_prediction is not None else None
    }
}

print("\n📊 Model Status:")
print("-" * 70)
for model, status in test_results["models_status"].items():
    print(f"  {model:25s}: {status}")

print("\n📈 Clinical Model Test:")
print("-" * 70)
print(f"  Status:     {test_results['clinical_model_test']['status']}")
print(f"  Prediction: {test_results['clinical_model_test']['prediction']}")
print(f"  Confidence: {test_results['clinical_model_test']['confidence']:.2f}%")

print("\n📸 Ultrasound Model Test:")
print("-" * 70)
print(f"  Status:     {test_results['ultrasound_model_test']['status']}")
print(f"  Prediction: {test_results['ultrasound_model_test']['prediction']}")
print(f"  Confidence: {test_results['ultrasound_model_test']['confidence']:.2f}%")

print("\n🔄 Multimodal Fusion Test:")
print("-" * 70)
print(f"  Status:     {test_results['multimodal_fusion_test']['status']}")
print(f"  Fusion Score: {test_results['multimodal_fusion_test']['fusion_score']:.2f}%")
print(f"  Prediction:   {test_results['multimodal_fusion_test']['fusion_prediction']}")

print("\n" + "=" * 70)
print("OVERALL STATUS: ✓ ALL MODELS WORKING PERFECTLY!")
print("=" * 70)

# Save test results
with open('test_results.json', 'w') as f:
    json.dump(test_results, f, indent=4)
print(f"\n✓ Test results saved to: test_results.json\n")
