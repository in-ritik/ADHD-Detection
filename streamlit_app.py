import streamlit as st
import pandas as pd
import numpy as np
import io
import sys
import os
import time

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.pred import load_and_merge_data, train_final_model, BEST_FEATURES, ID_COL, TARGET_COL
except ImportError:
    st.error("System Error: Source modules not found.")
    st.stop()

# Basic setup
st.set_page_config(
    page_title="ADHD Diagnostic Support",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Styling: Force light mode and clean typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Force light background everywhere */
    :root, html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stToolbar"], .stApp {
        background-color: #f8fafc !important; 
        color: #0f172a !important;
    }
    
    /* Input fields */
    .stTextInput input, .stSelectbox div[data-baseweb="select"], .stFileUploader {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6, span, div, p, label, .stMarkdown {
        color: #0f172a !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Secondary text */
    .stCaption, small, .small-text, .stText {
        color: #475569 !important;
    }

    /* Layout tweaks */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
        max-width: 95% !important;
    }

    /* Header sizing */
    h1 { font-size: 1.8rem !important; margin-bottom: 0.5rem !important; }
    h2 { font-size: 1.4rem !important; }
    h3 { font-size: 1.2rem !important; margin-top: 0 !important; }
    h4 { font-size: 1rem !important; font-weight: 500 !important; margin-top: 0 !important; }
    hr { margin: 1rem 0 !important; }

    /* Info box styling */
    .info-box {
        background-color: #ffffff !important;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #2563eb;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    .info-box p { margin-bottom: 0.5rem; font-size: 0.85rem; color: #334155 !important; }
    .info-box li { font-size: 0.85rem; color: #334155 !important; }

    /* Upload area */
    div[data-testid="stFileUploader"] {
        background-color: #ffffff !important;
        padding: 1rem;
        border: 2px dashed #94a3b8;
        min-height: auto;
    }
    div[data-testid="stFileUploader"] small {
        color: #64748b !important;
    }
    
    /* Results */
    .result-container {
        display: flex; 
        gap: 1rem; 
        margin-bottom: 1rem;
    }
    .result-card {
        background-color: white !important;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        text-align: center;
        flex: 1;
        border: 1px solid #e2e8f0;
    }
    .result-value { font-size: 1.5rem; font-weight: 800; }
    .result-label { font-size: 0.75rem; font-weight: 700; text-transform: uppercase; }

    /* Buttons */
    .stButton > button {
        background-color: #0f172a !important;
        color: white !important;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton > button:hover {
        background-color: #334155 !important;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("ADHD Diagnostic Support")
st.markdown("#### Clinical Assessment Analysis")
st.markdown("---")

# Load and train model
@st.cache_resource
def get_model():
    try:
        X, y = load_and_merge_data(feature_list=BEST_FEATURES)
        model = train_final_model(X, y)
    except Exception as e:
        return None, str(e)
    return model, None

model, error_msg = get_model()
if error_msg:
    st.error(f"Initialization Failed: {error_msg}")
    st.stop()

# Two-column layout
left_col, right_col = st.columns([1.2, 2], gap="medium")

with left_col:
    # Context
    st.markdown("""
    <div class="info-box">
        <h3>Methodology</h3>
        <p><strong>Objective:</strong> Adjunctive tool for ADHD evaluation complementing qualitative assessment.</p>
        <p><strong>Model:</strong> Logistic Regression on <strong>75 biomarkers</strong> from CPT-II (Continuous Performance Test) & Demographics.</p>
        <p><strong>Metrics Analyzed:</strong></p>
        <ul style="margin-bottom:0; padding-left:1.2rem;">
            <li>Omission & Commission Errors</li>
            <li>Reaction Time Variability (Hit RT)</li>
            <li>Signal Entropy & Fourier Coefficients</li>
        </ul>
        <p style="margin-top:0.5rem; font-style:italic; font-size:0.8rem;">Note: AUC ≥ 0.98. Not for standalone diagnosis.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Upload Record")
    uploaded_file = st.file_uploader("CPT-II CSV Record", type=["csv"], label_visibility="collapsed")


with right_col:
    if uploaded_file is None:
        st.info("Please upload a patient CSV file to generate analysis.")
        
        # Empty state
        st.markdown("""
        <div style="text-align: center; color: #64748b; padding: 3rem; border: 2px dashed #e2e8f0; border-radius: 12px;">
            <h4>Waiting for Data...</h4>
        </div>
        """, unsafe_allow_html=True)
    
    elif model is not None:
        try:
            content = uploaded_file.read()
            try:
                df = pd.read_csv(io.BytesIO(content), delimiter=',')
                if len(df.columns) < 5: df = pd.read_csv(io.BytesIO(content), delimiter=';')
            except: df = pd.read_csv(io.BytesIO(content), delimiter=';')

            missing_cols = [col for col in BEST_FEATURES if col not in df.columns]
            
            if missing_cols:
                st.error("Invalid Schema. Missing required feature columns.")
            else:
                patient_row = df.iloc[[0]]
                features = patient_row[BEST_FEATURES]
                prob = model.predict_proba(features)[:, 1][0]
                prediction_class = int(model.predict(features)[0])
                
                # Colors
                if prediction_class == 1:
                    color_main = "#D32F2F"; color_bg = "#FFEBEE"; class_label = "ADHD POSITIVE"; bar_color = "#D32F2F"
                else:
                    color_main = "#1B5E20"; color_bg = "#E8F5E9"; class_label = "ADHD NEGATIVE"; bar_color = "#2E7D32"

                # Results
                st.markdown(f"""
                <div class="result-container">
                    <div class="result-card" style="background-color: {color_bg} !important; border-color: {color_main}; flex: 1.5;">
                        <div class="result-label" style="color: {color_main} !important;">Classification</div>
                        <div class="result-value" style="color: {color_main} !important;">{class_label}</div>
                    </div>
                    <div class="result-card" style="flex: 1;">
                        <div class="result-label">Probability</div>
                        <div class="result-value">{prob:.1%}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence bar
                st.markdown("**Confidence Assessment**")
                st.markdown(f"""
                <div style="width: 100%; background-color: #e2e8f0; border-radius: 6px; height: 1.2rem; overflow: hidden; margin-top: 0.2rem; border: 1px solid #cbd5e1;">
                    <div style="width: {prob * 100}%; background-color: {bar_color}; height: 100%; transition: width 0.5s ease;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #475569; margin-top: 0.1rem;">
                    <span>0%</span><span>Risk Scale</span><span>100%</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Validation (Compact)
                if TARGET_COL in patient_row.columns:
                    st.markdown("---")
                    actual = int(patient_row[TARGET_COL].values[0])
                    matches = (prediction_class == actual)
                    gt_label = 'Positive' if actual == 1 else 'Negative'
                    
                    if matches:
                        st.markdown(f"<span style='color: #166534; font-weight: 600;'>Validated against Ground Truth ({gt_label})</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<span style='color: #991b1b; font-weight: 600;'>Discordant with Ground Truth ({gt_label})</span>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {str(e)}")
