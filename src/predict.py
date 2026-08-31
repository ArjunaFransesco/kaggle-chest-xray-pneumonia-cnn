# -*- coding: utf-8 -*-
"""
Batch Inference & Clinical Threshold Calibration Engine
Chest Radiograph Pneumonia Diagnostic Engine
Author: Arjuna Fransesco (https://github.com/ArjunaFransesco)
"""

import os
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "model_pipeline.joblib")
FEATURE_NAMES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "feature_names.joblib")


class ChestXRayInferenceEngine:
    def __init__(self, model_path=MODEL_PATH, feature_path=FEATURE_NAMES_PATH):
        self.model_path = model_path
        self.feature_path = feature_path
        self.model = None
        self.features = None
        self._load_artifacts()

    def _load_artifacts(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        if os.path.exists(self.feature_path):
            self.features = joblib.load(self.feature_path)

    def predict_single(self, feature_dict, threshold=0.50):
        """
        Run clinical prediction on a single chest x-ray telemetry vector.
        """
        df = pd.DataFrame([feature_dict])
        if self.features:
            df = df[self.features]

        prob = self.model.predict_proba(df)[0][1] if hasattr(self.model, "predict_proba") else 0.5
        prediction = 1 if prob >= threshold else 0
        label = "PNEUMONIA" if prediction == 1 else "NORMAL"

        return {
            "label": label,
            "prediction": int(prediction),
            "pneumonia_probability": float(prob),
            "confidence": float(max(prob, 1 - prob) * 100),
            "risk_tier": "HIGH RISK" if prob > 0.70 else ("MODERATE RISK" if prob > 0.35 else "LOW RISK")
        }

    def predict_batch(self, df_features, threshold=0.50):
        """
        Execute vectorized batch inference across multiple patient records.
        """
        if self.features:
            df_aligned = df_features[self.features]
        else:
            df_aligned = df_features

        probs = self.model.predict_proba(df_aligned)[:, 1] if hasattr(self.model, "predict_proba") else np.zeros(len(df_features))
        preds = (probs >= threshold).astype(int)

        results = df_features.copy()
        results["pneumonia_prob"] = probs
        results["predicted_class"] = preds
        results["predicted_label"] = np.where(preds == 1, "PNEUMONIA", "NORMAL")
        return results


if __name__ == "__main__":
    engine = ChestXRayInferenceEngine()
    sample = {
        "mean_pixel_intensity": 115.4,
        "std_pixel_intensity": 48.2,
        "entropy": 6.82,
        "contrast_glcm": 28.4,
        "opacity_density_ratio": 0.42,
        "lung_field_symmetry": 0.88,
        "consolidation_score": 0.65,
        "bronchial_prominence": 0.55
    }
    res = engine.predict_single(sample)
    print(f"Sample Diagnosis: {res['label']} ({res['confidence']:.2f}% confidence, Risk: {res['risk_tier']})")
