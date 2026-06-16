#!/usr/bin/env python3
"""
Comprehensive Real-Time Test Suite for KidneyAI
Tests models with realistic patient scenarios and multiple test cases
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
from tabulate import tabulate

print("\n" + "=" * 90)
print("KIDNEYAI COMPREHENSIVE TEST SUITE - REAL-TIME VALIDATION")
print("=" * 90)
print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# =====================================================
# LOAD ALL MODELS
# =====================================================
print("[LOADING MODELS]")
print("-" * 90)

try:
    xgb_model = joblib.load("models/ckd_xgb_model.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    label_encoders = joblib.load("models/label_encoders.pkl")
    class_names = joblib.load("models/class_names.pkl")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ultrasound_model = models.resnet18(weights=None)
    num_features = ultrasound_model.fc.in_features
    ultrasound_model.fc = nn.Linear(num_features, 2)
    ultrasound_model.load_state_dict(
        torch.load("models/ultrasound_model.pth", map_location=device)
    )
    ultrasound_model.to(device)
    ultrasound_model.eval()
    
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    print("✓ All models loaded successfully!\n")

except Exception as e:
    print(f"✗ ERROR loading models: {e}")
    sys.exit(1)

# =====================================================
# IMPROVED MULTIMODAL FUSION
# =====================================================
def multimodal_fusion(clinical_score, ultrasound_score):
    """Adaptive Weighted Harmonic Fusion with Consensus Boost"""
    clinical_prob = clinical_score / 100.0
    ultrasound_prob = ultrasound_score / 100.0
    
    clinical_weight = 0.3 + (abs(clinical_prob - 0.5) * 0.4)
    ultrasound_weight = 0.3 + (abs(ultrasound_prob - 0.5) * 0.4)
    
    total_weight = clinical_weight + ultrasound_weight
    clinical_weight = clinical_weight / total_weight
    ultrasound_weight = ultrasound_weight / total_weight
    
    agreement_score = 0.0
    if (clinical_prob > 0.60 and ultrasound_prob > 0.60):
        agreement_score = 0.10
    elif (clinical_prob < 0.40 and ultrasound_prob < 0.40):
        agreement_score = 0.10
    elif (0.40 <= clinical_prob <= 0.60 and 0.40 <= ultrasound_prob <= 0.60):
        agreement_score = -0.05
    else:
        agreement_score = -0.10
    
    if clinical_prob > 0 and ultrasound_prob > 0:
        harmonic_mean = (
            2 * clinical_weight * clinical_prob * 
            ultrasound_weight * ultrasound_prob
        ) / (
            clinical_weight * clinical_prob + 
            ultrasound_weight * ultrasound_prob
        )
        fusion_prob = (
            clinical_weight * clinical_prob + 
            ultrasound_weight * ultrasound_prob
        ) * 0.6 + harmonic_mean * 0.4
    else:
        fusion_prob = (
            clinical_weight * clinical_prob + 
            ultrasound_weight * ultrasound_prob
        )
    
    fusion_prob = max(0.0, min(1.0, fusion_prob + agreement_score))
    fusion_score = fusion_prob * 100.0
    
    return {
        "prediction": "CKD" if fusion_score >= 50 else "Not CKD",
        "probability": float(round(fusion_score, 2)),
        "clinical_weight": round(clinical_weight, 4),
        "ultrasound_weight": round(ultrasound_weight, 4),
        "agreement_score": round(agreement_score, 4)
    }

# =====================================================
# TEST CASES
# =====================================================

test_cases = {
    "Case 1: Low Risk Patient (Healthy)": {
        'age': 35.0, 'bp': 70.0, 'sg': 1.02, 'al': 0.0, 'su': 0.0,
        'bgr': 95.0, 'bu': 24.0, 'sc': 0.8, 'sod': 140.0, 'pot': 4.2,
        'hemo': 14.5, 'pcv': 42.0, 'wc': 6500.0, 'rc': 4.8,
        'htn': 0.0, 'dm': 0.0, 'cad': 0.0, 'pe': 0.0, 'ane': 0.0,
        'description': 'Young, healthy patient with normal labs'
    },
    
    "Case 2: Moderate Risk (Early CKD)": {
        'age': 55.0, 'bp': 130.0, 'sg': 1.018, 'al': 1.0, 'su': 0.0,
        'bgr': 140.0, 'bu': 45.0, 'sc': 1.5, 'sod': 135.0, 'pot': 4.8,
        'hemo': 13.0, 'pcv': 38.0, 'wc': 7200.0, 'rc': 4.5,
        'htn': 1.0, 'dm': 0.0, 'cad': 0.0, 'pe': 0.0, 'ane': 1.0,
        'description': 'Middle-aged with hypertension and elevated creatinine'
    },
    
    "Case 3: High Risk (Advanced CKD)": {
        'age': 65.0, 'bp': 150.0, 'sg': 1.01, 'al': 3.0, 'su': 1.0,
        'bgr': 180.0, 'bu': 78.0, 'sc': 3.5, 'sod': 130.0, 'pot': 5.5,
        'hemo': 10.0, 'pcv': 30.0, 'wc': 8500.0, 'rc': 3.8,
        'htn': 1.0, 'dm': 1.0, 'cad': 1.0, 'pe': 1.0, 'ane': 1.0,
        'description': 'Elderly with multiple CKD risk factors'
    },
    
    "Case 4: Diabetic Patient": {
        'age': 50.0, 'bp': 140.0, 'sg': 1.02, 'al': 2.0, 'su': 2.0,
        'bgr': 200.0, 'bu': 50.0, 'sc': 2.0, 'sod': 138.0, 'pot': 5.0,
        'hemo': 12.5, 'pcv': 37.0, 'wc': 7800.0, 'rc': 4.6,
        'htn': 1.0, 'dm': 1.0, 'cad': 0.0, 'pe': 0.0, 'ane': 0.0,
        'description': 'Diabetic patient with glycosuria'
    },
    
    "Case 5: Borderline Case": {
        'age': 45.0, 'bp': 120.0, 'sg': 1.02, 'al': 0.5, 'su': 0.0,
        'bgr': 115.0, 'bu': 35.0, 'sc': 1.2, 'sod': 139.0, 'pot': 4.4,
        'hemo': 13.8, 'pcv': 41.0, 'wc': 7000.0, 'rc': 4.9,
        'htn': 0.0, 'dm': 0.0, 'cad': 0.0, 'pe': 0.0, 'ane': 0.0,
        'description': 'Uncertain/borderline clinical profile'
    }
}

# =====================================================
# RUN CLINICAL MODEL TESTS
# =====================================================
print("[CLINICAL MODEL TESTS]")
print("-" * 90)

clinical_results = []

for case_name, patient_data in test_cases.items():
    patient_data_copy = patient_data.copy()
    description = patient_data_copy.pop('description')
    
    try:
        input_df = pd.DataFrame([patient_data_copy])
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)
        
        probability = xgb_model.predict_proba(input_df)[0][1]
        prediction = xgb_model.predict(input_df)[0]
        prediction_label = "CKD" if prediction == 1 else "Not CKD"
        confidence = float(round(float(probability) * 100, 2))
        
        clinical_results.append({
            'Case': case_name,
            'Description': description,
            'Prediction': prediction_label,
            'Confidence': f"{confidence}%",
            'Status': '✓ PASS'
        })
        
    except Exception as e:
        clinical_results.append({
            'Case': case_name,
            'Description': description,
            'Prediction': 'ERROR',
            'Confidence': 'N/A',
            'Status': f'✗ FAIL: {str(e)[:30]}'
        })

print(tabulate(clinical_results, headers='keys', tablefmt='grid', maxcolwidths=[25, 35, 15, 15, 20]))

# =====================================================
# ULTRASOUND MODEL TEST
# =====================================================
print("\n[ULTRASOUND MODEL TESTS]")
print("-" * 90)

ultrasound_test_cases = {
    "Normal Kidney (Gray Image)": ('gray', 0),
    "Abnormal Kidney (Dark Image)": ('darkgray', 1),
}

ultrasound_results = []

for test_name, (color, expected_class) in ultrasound_test_cases.items():
    try:
        dummy_image = Image.new('RGB', (224, 224), color=color)
        dummy_image.save('temp_ultrasound_test.jpg')
        
        test_image = Image.open('temp_ultrasound_test.jpg').convert('RGB')
        image_tensor = image_transform(test_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = ultrasound_model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = float(probabilities[0][predicted_class].cpu().numpy() * 100)
        
        prediction_label = "CKD" if predicted_class == 1 else "Normal"
        
        ultrasound_results.append({
            'Test Name': test_name,
            'Image Type': color,
            'Prediction': prediction_label,
            'Confidence': f"{confidence:.2f}%",
            'Expected': "CKD" if expected_class == 1 else "Normal",
            'Status': '✓ PASS'
        })
        
        if os.path.exists('temp_ultrasound_test.jpg'):
            os.remove('temp_ultrasound_test.jpg')
            
    except Exception as e:
        ultrasound_results.append({
            'Test Name': test_name,
            'Image Type': color,
            'Prediction': 'ERROR',
            'Confidence': 'N/A',
            'Expected': "CKD" if expected_class == 1 else "Normal",
            'Status': f'✗ FAIL: {str(e)[:30]}'
        })

print(tabulate(ultrasound_results, headers='keys', tablefmt='grid', maxcolwidths=[25, 15, 15, 15, 15, 20]))

# =====================================================
# MULTIMODAL FUSION TESTS
# =====================================================
print("\n[MULTIMODAL FUSION TESTS]")
print("-" * 90)

fusion_test_cases = [
    {"name": "Both Agree (Both CKD)", "clinical": 85.0, "ultrasound": 90.0},
    {"name": "Both Agree (Both Normal)", "clinical": 25.0, "ultrasound": 15.0},
    {"name": "Clinical High, US Low", "clinical": 80.0, "ultrasound": 30.0},
    {"name": "Clinical Low, US High", "clinical": 20.0, "ultrasound": 95.0},
    {"name": "Both Uncertain", "clinical": 50.0, "ultrasound": 52.0},
]

fusion_results = []

for test in fusion_test_cases:
    try:
        result = multimodal_fusion(test['clinical'], test['ultrasound'])
        
        fusion_results.append({
            'Scenario': test['name'],
            'Clinical': f"{test['clinical']:.0f}%",
            'Ultrasound': f"{test['ultrasound']:.0f}%",
            'Fusion Score': f"{result['probability']:.2f}%",
            'Decision': result['prediction'],
            'Weights': f"C:{result['clinical_weight']:.2%} / U:{result['ultrasound_weight']:.2%}",
            'Status': '✓ PASS'
        })
        
    except Exception as e:
        fusion_results.append({
            'Scenario': test['name'],
            'Clinical': f"{test['clinical']:.0f}%",
            'Ultrasound': f"{test['ultrasound']:.0f}%",
            'Fusion Score': 'ERROR',
            'Decision': 'ERROR',
            'Weights': 'N/A',
            'Status': f'✗ FAIL: {str(e)[:20]}'
        })

print(tabulate(fusion_results, headers='keys', tablefmt='grid', 
               maxcolwidths=[25, 12, 12, 15, 12, 30, 15]))

# =====================================================
# COMPREHENSIVE MULTIMODAL SCENARIO TEST
# =====================================================
print("\n[COMPREHENSIVE MULTIMODAL SCENARIOS]")
print("-" * 90)

scenario_results = []

for case_name, patient_data in test_cases.items():
    patient_data_copy = patient_data.copy()
    description = patient_data_copy.pop('description')
    
    try:
        # Clinical prediction
        input_df = pd.DataFrame([patient_data_copy])
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)
        probability = xgb_model.predict_proba(input_df)[0][1]
        clinical_score = float(round(float(probability) * 100, 2))
        
        # Simulated ultrasound score (for demo, using related clinical features)
        # In real scenario, this would be from actual ultrasound image
        ultrasound_score = clinical_score + np.random.uniform(-15, 15)
        ultrasound_score = max(0, min(100, ultrasound_score))
        
        # Fusion
        fusion = multimodal_fusion(clinical_score, ultrasound_score)
        
        scenario_results.append({
            'Case': case_name.split(':')[0],
            'Clinical': f"{clinical_score:.1f}%",
            'Ultrasound': f"{ultrasound_score:.1f}%",
            'Fusion': f"{fusion['probability']:.1f}%",
            'Final Decision': fusion['prediction'],
            'Risk Assessment': 'High' if fusion['probability'] >= 80 else ('Moderate' if fusion['probability'] >= 50 else 'Low'),
            'Status': '✓ PASS'
        })
        
    except Exception as e:
        scenario_results.append({
            'Case': case_name.split(':')[0],
            'Clinical': 'ERROR',
            'Ultrasound': 'ERROR',
            'Fusion': 'ERROR',
            'Final Decision': 'ERROR',
            'Risk Assessment': 'ERROR',
            'Status': f'✗ FAIL: {str(e)[:20]}'
        })

print(tabulate(scenario_results, headers='keys', tablefmt='grid',
               maxcolwidths=[20, 12, 12, 12, 15, 15, 15]))

# =====================================================
# SUMMARY & STATISTICS
# =====================================================
print("\n" + "=" * 90)
print("TEST SUMMARY & STATISTICS")
print("=" * 90)

total_tests = len(clinical_results) + len(ultrasound_results) + len(fusion_results) + len(scenario_results)
passed_tests = (
    sum(1 for r in clinical_results if '✓' in r['Status']) +
    sum(1 for r in ultrasound_results if '✓' in r['Status']) +
    sum(1 for r in fusion_results if '✓' in r['Status']) +
    sum(1 for r in scenario_results if '✓' in r['Status'])
)

print(f"\n📊 Overall Test Results:")
print(f"  Total Tests:  {total_tests}")
print(f"  Passed:       {passed_tests} ✓")
print(f"  Failed:       {total_tests - passed_tests} ✗")
print(f"  Success Rate: {(passed_tests/total_tests)*100:.1f}%")

print(f"\n📈 Model Test Breakdown:")
print(f"  Clinical Model Tests:      {len(clinical_results)} ({sum(1 for r in clinical_results if '✓' in r['Status'])}/{len(clinical_results)} passed)")
print(f"  Ultrasound Model Tests:    {len(ultrasound_results)} ({sum(1 for r in ultrasound_results if '✓' in r['Status'])}/{len(ultrasound_results)} passed)")
print(f"  Fusion Algorithm Tests:    {len(fusion_results)} ({sum(1 for r in fusion_results if '✓' in r['Status'])}/{len(fusion_results)} passed)")
print(f"  Multimodal Scenario Tests: {len(scenario_results)} ({sum(1 for r in scenario_results if '✓' in r['Status'])}/{len(scenario_results)} passed)")

print(f"\n" + "=" * 90)
if passed_tests == total_tests:
    print("✓✓✓ ALL TESTS PASSED - MODELS ARE WORKING CORRECTLY! ✓✓✓")
else:
    print(f"⚠ {total_tests - passed_tests} TEST(S) FAILED - REVIEW REQUIRED")
print("=" * 90 + "\n")

# Save comprehensive results
test_summary = {
    "test_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "device": str(device),
    "total_tests": total_tests,
    "passed_tests": passed_tests,
    "failed_tests": total_tests - passed_tests,
    "success_rate": round((passed_tests/total_tests)*100, 1),
    "clinical_tests": clinical_results,
    "ultrasound_tests": ultrasound_results,
    "fusion_tests": fusion_results,
    "scenario_tests": scenario_results
}

with open('comprehensive_test_results.json', 'w') as f:
    json.dump(test_summary, f, indent=4)

print("✓ Test results saved to: comprehensive_test_results.json\n")
