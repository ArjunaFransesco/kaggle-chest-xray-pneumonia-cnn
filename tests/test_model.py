# -*- coding: utf-8 -*-
"""
Automated Unit Tests for Chest X-Ray Diagnostic Engine
Author: Arjuna Fransesco (https://github.com/ArjunaFransesco)
"""

import os
import unittest
import numpy as np
import pandas as pd
from src.predict import ChestXRayInferenceEngine


class TestChestXRayModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = ChestXRayInferenceEngine()

    def test_single_prediction_normal_sample(self):
        sample_normal = {
            "patient_age_years": 28.0,
            "biomarker_alpha_level": 1.2,
            "radiological_opacity_density": 0.12,
            "inflammatory_crp_mg_l": 2.5,
            "oxygen_saturation_pct": 99.0,
            "comorbidity_index": 0
        }
        res = self.engine.predict_single(sample_normal)
        self.assertIn("label", res)
        self.assertIn("pneumonia_probability", res)
        self.assertGreaterEqual(res["pneumonia_probability"], 0.0)
        self.assertLessEqual(res["pneumonia_probability"], 1.0)
        self.assertIn(res["label"], ["NORMAL_CLEAR", "PNEUMONIA_DETECTED"])

    def test_single_prediction_pneumonia_sample(self):
        sample_pneumonia = {
            "patient_age_years": 72.0,
            "biomarker_alpha_level": 7.8,
            "radiological_opacity_density": 0.88,
            "inflammatory_crp_mg_l": 48.0,
            "oxygen_saturation_pct": 86.0,
            "comorbidity_index": 4
        }
        res = self.engine.predict_single(sample_pneumonia)
        self.assertIn("confidence", res)
        self.assertGreaterEqual(res["confidence"], 50.0)
        self.assertIn(res["risk_tier"], ["HIGH CRITICAL", "MODERATE ELEVATED", "LOW NORMAL"])

    def test_batch_prediction(self):
        df = pd.DataFrame([
            {
                "patient_age_years": 25.0, "biomarker_alpha_level": 1.0, "radiological_opacity_density": 0.10,
                "inflammatory_crp_mg_l": 1.5, "oxygen_saturation_pct": 98.5, "comorbidity_index": 0
            },
            {
                "patient_age_years": 80.0, "biomarker_alpha_level": 8.5, "radiological_opacity_density": 0.92,
                "inflammatory_crp_mg_l": 55.0, "oxygen_saturation_pct": 84.0, "comorbidity_index": 5
            }
        ])
        results = self.engine.predict_batch(df)
        self.assertEqual(len(results), 2)
        self.assertIn("pneumonia_probability", results.columns)
        self.assertIn("predicted_label", results.columns)


if __name__ == "__main__":
    unittest.main()
