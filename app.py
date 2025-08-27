import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import os
from datetime import datetime

# ------------------ Helpers ------------------
def load_model_and_scaler():
    model = joblib.load("best_fire_detection_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

def map_confidence(conf):
    return {"low": 0, "nominal": 1, "high": 2}.get(conf, 1)

def predict(input_df, model, scaler):
    scaled = scaler.transform(input_df)
    pred = model.predict(scaled)
    return pred

def fire_label(idx):
    return {
        0: "Vegetation Fire",
        2: "Other Static Land Source",
        3: "Offshore Fire"
    }.get(idx, "Unknown")

def plot_feature_distributions(df):
    numeric = ["brightness", "bright_t31", "frp", "scan", "track"]
    for col in numeric:
        fig, ax = plt.subplots()
        ax.hist(df[col].dropna(), bins=30)
        ax.set_title(f"Distribution of {col.capitalize()}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

def show_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[fire_label(l) for l in labels])
    fig, ax = plt.subplots(figsize=(5,5))
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

# ------------------ App ------------------
st.set_page_config(page_title="Fire Type Classifier", layout="wide", initial_sidebar_state="expanded")

# Sidebar
st.sidebar.header("🔥 Fire Classification Project")
st.sidebar.markdown(
    """
    **Features**  
    - Upload MODIS fire data  
    - View exploratory analysis  
    - Make real-time predictions  
    - See model performance  
    - Export results  
    """
)
st.sidebar.markdown("---")
st.sidebar.subheader("Quick Input")
brightness = st.sidebar.number_input("Brightness", value=300.0, step=1.0)
bright_t31 = st.sidebar.number_input("Brightness T31", value=290.0, step=1.0)
frp = st.sidebar.number_input("Fire Radiative Power (FRP)", value=15.0, step=0.1)
scan = st.sidebar.number_input("Scan", value=1.0, step=0.1)
track = st.sidebar.number_input("Track", value=1.0, step=0.1)
confidence = st.sidebar.selectbox("Confidence Level", ["low", "nominal", "high"])
st.sidebar.markdown("---")

# Load model
try:
    model, scaler = load_model_and_scaler()
except Exception as e:
    st.error(f"Failed to load model/scaler: {e}")
    st.stop()

# Main layout
col1, col2 = st.columns([2, 3])

with col1:
    st.markdown("## 🔍 Prediction Panel")
    st.markdown("Enter values manually or upload a CSV for batch classification.")
    input_df = pd.DataFrame([[brightness, bright_t31, frp, scan, track, map_confidence(confidence)]],
                            columns=["brightness", "bright_t31", "frp", "scan", "track", "confidence"])

    st.subheader("Manual Input")
    st.dataframe(input_df)

    if st.button("Predict Fire Type"):
        pred_idx = predict(input_df, model, scaler)[0]
        result = fire_label(pred_idx)
        st.success(f"**Predicted Fire Type:** {result}")

        # Save history
        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "brightness": brightness,
            "bright_t31": bright_t31,
            "frp": frp,
            "scan": scan,
            "track": track,
            "confidence": confidence,
            "prediction": result
        })

    st.subheader("Prediction History")
    if "history" in st.session_state and st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df)
        st.download_button("Download History as CSV", hist_df.to_csv(index=False), "history.csv")

with col2:
    st.markdown("## 📊 Data & Analysis")
    uploaded = st.file_uploader("Upload MODIS Fire Data CSV", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.subheader("Raw Data Preview")
            st.dataframe(df.head(10))

            # Basic EDA
            st.subheader("Feature Distributions")
            plot_feature_distributions(df)

            st.subheader("Sample Predictions (Batch)")
            if st.button("Run Batch Prediction on Uploaded Data"):
                required_cols = ["brightness", "bright_t31", "frp", "scan", "track", "confidence"]
                if not all(col in df.columns for col in required_cols):
                    st.warning(f"CSV must contain columns: {required_cols}")
                else:
                    df_proc = df[required_cols].copy()
                    df_proc["confidence"] = df_proc["confidence"].map({"low": 0, "nominal": 1, "high": 2})
                    preds = predict(df_proc.values, model, scaler)
                    df["predicted_fire_type"] = [fire_label(p) for p in preds]
                    st.dataframe(df[["brightness", "frp", "confidence", "predicted_fire_type"]].head(15))
                    st.download_button("Download Predictions", df.to_csv(index=False), "batch_predictions.csv")
        except Exception as e:
            st.error(f"Failed to process uploaded file: {e}")
    else:
        st.info("Upload a CSV with MODIS features to explore and batch-predict.")

st.markdown("---")
st.markdown("## 🧠 Model Info & Notes")
st.write(
    """
    - **Input Features:** brightness, bright_t31, frp, scan, track, confidence (mapped to numeric).  
    - **Fire Types Classified:** Vegetation Fire, Other Static Land Source, Offshore Fire.  
    - **Backend:** Scaled inputs via StandardScaler, model persisted with joblib.  
    - **Interface:** Manual and batch prediction support.  
    """
)

# Optional: add model evaluation if labels provided in upload
if uploaded:
    if 'true_label' in locals() or ('df' in locals() and 'true_fire_type' in df.columns and 'predicted_fire_type' in df.columns):
        st.subheader("Model Evaluation (if ground truth available)")
        # infer ground truth and predicted
        if 'df' in locals() and 'true_fire_type' in df.columns:
            # map true labels to indices if needed (assumes same mapping as model)
            label_map = {"Vegetation Fire": 0, "Other Static Land Source": 2, "Offshore Fire": 3}
            y_true = df["true_fire_type"].map(label_map).dropna().astype(int)
            y_pred = [list(label_map.values())[list(label_map.keys()).index(pt)] if pt in label_map else -1 for pt in df["predicted_fire_type"]]
            if len(y_true) == len(y_pred):
                st.text("Classification Report:")
                st.text(classification_report(y_true, y_pred, zero_division=0))
                show_confusion_matrix(y_true, y_pred, labels=[0,2,3])
            else:
                st.warning("Ground truth and prediction lengths mismatch.")

st.markdown("### ⚙️ Tips for Presentation")
st.markdown(
    """
    - Take screenshot of prediction panel with a sample result.  
    - Include a small flow diagram: Data → Preprocess → Model → Web App.  
    - Use exported history/predictions as appendix in your slides.  
    """
)
