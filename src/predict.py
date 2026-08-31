# -*- coding: utf-8 -*-
"""
Clinical Batch & Real-Time Inference Engine
Chest Radiograph Pneumonia Diagnostic Engine
Author: Arjuna Fransesco (https://github.com/ArjunaFransesco)
"""

import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_pipeline.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.joblib")
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, "models", "feature_names.joblib")


class ChestXRayInferenceEngine:
    def __init__(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH, feature_path=FEATURE_NAMES_PATH):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.feature_path = feature_path
        self.model = None
        self.scaler = None
        self.features = None
        self._load_artifacts()

    def _load_artifacts(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
        if os.path.exists(self.feature_path):
            self.features = joblib.load(self.feature_path)

    def predict_single(self, feature_dict, threshold=0.50):
        """
        Run clinical prediction on a single chest x-ray biomarker vector.
        """
        df = pd.DataFrame([feature_dict])
        if self.features:
            df = df[self.features]

        X_scaled = self.scaler.transform(df) if self.scaler else df
        prob = self.model.predict_proba(X_scaled)[0][1] if hasattr(self.model, "predict_proba") else 0.5
        prediction = 1 if prob >= threshold else 0
        label = "PNEUMONIA_DETECTED" if prediction == 1 else "NORMAL_CLEAR"

        return {
            "label": label,
            "prediction": int(prediction),
            "pneumonia_probability": float(prob),
            "confidence": float(max(prob, 1 - prob) * 100),
            "risk_tier": "HIGH CRITICAL" if prob > 0.70 else ("MODERATE ELEVATED" if prob > 0.35 else "LOW NORMAL")
        }

    def predict_batch(self, df_features, threshold=0.50):
        """
        Execute vectorized batch inference across multiple patient records.
        """
        if self.features:
            df_aligned = df_features[self.features]
        else:
            df_aligned = df_features

        X_scaled = self.scaler.transform(df_aligned) if self.scaler else df_aligned
        probs = self.model.predict_proba(X_scaled)[:, 1] if hasattr(self.model, "predict_proba") else np.zeros(len(df_features))
        preds = (probs >= threshold).astype(int)

        results = df_features.copy()
        results["pneumonia_probability"] = probs
        results["predicted_class"] = preds
        results["predicted_label"] = np.where(preds == 1, "PNEUMONIA_DETECTED", "NORMAL_CLEAR")
        return results


if __name__ == "__main__":
    engine = ChestXRayInferenceEngine()
    sample = {
        "patient_age_years": 58.0,
        "biomarker_alpha_level": 4.2,
        "radiological_opacity_density": 0.68,
        "inflammatory_crp_mg_l": 24.5,
        "oxygen_saturation_pct": 92.0,
        "comorbidity_index": 2
    }
    res = engine.predict_single(sample)
    print(f"Sample Diagnosis: {res['label']} ({res['confidence']:.2f}% confidence, Risk: {res['risk_tier']})")
