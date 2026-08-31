import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Chest Radiograph Pneumonia Diagnostic Engine & Feature Visualizer",
    page_icon="🤖",
    layout="wide"
)

st.title("🎯 Chest Radiograph Pneumonia Diagnostic Engine & Feature Visualizer")
st.markdown("**Domain**: `Deep Learning / Medical Imaging` | **Tech Stack**: `PyTorch CNN, Scikit-Learn, Streamlit`")
st.markdown("**Author**: [Arjuna Fransesco](https://github.com/ArjunaFransesco) | **GitHub**: [Portfolio Repositories](https://github.com/ArjunaFransesco?tab=repositories)")
st.markdown("---")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("⚙️ Domain Input Telemetry")
    patient_age_years = st.slider("Patient Age Years", float(18.0), float(92.0), float(54.0))
    biomarker_alpha_level = st.slider("Biomarker Alpha Level", float(0.5), float(9.5), float(3.8))
    radiological_opacity_density = st.slider("Radiological Opacity Density", float(0.05), float(0.98), float(0.45))
    inflammatory_crp_mg_l = st.slider("Inflammatory Crp Mg L", float(0.5), float(60.0), float(12.5))
    oxygen_saturation_pct = st.slider("Oxygen Saturation Pct", float(80.0), float(100.0), float(96.0))
    comorbidity_index = st.slider("Comorbidity Index", int(0), int(5), int(2))

with col2:
    st.subheader("🔮 Predictive Model Inference")
    model_path = os.path.join(os.path.dirname(__file__), "models/model_pipeline.joblib")
    scaler_path = os.path.join(os.path.dirname(__file__), "models/scaler.joblib")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        input_df = pd.DataFrame([{"patient_age_years": patient_age_years, "biomarker_alpha_level": biomarker_alpha_level, "radiological_opacity_density": radiological_opacity_density, "inflammatory_crp_mg_l": inflammatory_crp_mg_l, "oxygen_saturation_pct": oxygen_saturation_pct, "comorbidity_index": comorbidity_index}])
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        
        st.markdown("#### Real-Time Prediction Output")
        st.info(f"Predicted `pathology_detected`: **{pred}**")
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_scaled)[0]
            st.progress(float(probs[1]) if len(probs) > 1 else float(probs[0]))
            st.caption(f"Confidence Probability Score: **{np.max(probs):.2%}**")
    else:
        st.warning("Model or Scaler artifact not found in models/ directory.")

st.markdown("---")
st.markdown("### 📊 Benchmark Metrics")
metrics_path = os.path.join(os.path.dirname(__file__), "reports/metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
    st.json(metrics_data)
