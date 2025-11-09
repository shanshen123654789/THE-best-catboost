import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor

# Set matplotlib parameters
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# Initialize application state
def initialize_application():
    """
    Initialize application state
    """
    # Application state
    if 'app_initialized' not in st.session_state:
        st.session_state.app_initialized = True
    
    # Model state
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    # Prediction state
    if 'predicted_value' not in st.session_state:
        st.session_state.predicted_value = None
    if 'shap_values' not in st.session_state:
        st.session_state.shap_values = None
    if 'explainer' not in st.session_state:
        st.session_state.explainer = None
    
    # Feature state
    if 'feature_values' not in st.session_state:
        st.session_state.feature_values = {}
    if 'features_df' not in st.session_state:
        st.session_state.features_df = None

# Initialize application
initialize_application()

# Page configuration
st.set_page_config(
    page_title="Growth Rate Prediction Model",
    page_icon="🐷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model automatically (removed manual load button)
@st.cache_resource
def load_model():
    """
    Load machine learning model with caching to avoid reloading
    """
    try:
        # Try loading with joblib
        model = joblib.load('catboost.pkl')
        return model, "joblib", True
    except:
        try:
            # If joblib fails, try CatBoost native format
            model = CatBoostRegressor()
            model.load_model('catboost.cbm')
            return model, "catboost", True
        except Exception as e:
            return None, str(e), False

# Auto-load model when app starts
if not st.session_state.model_loaded:
    with st.spinner('Loading prediction model...'):
        model, status, success = load_model()
        if success:
            st.session_state.model = model
            st.session_state.model_loaded = True
            
            # Get model feature order
            try:
                if hasattr(model, 'feature_names_'):
                    st.session_state.model_feature_names = model.feature_names_
                else:
                    # Infer order based on common patterns
                    st.session_state.model_feature_names = ['Sex', '30kg ABW', 'Litter size', 'Season', 'Birth weight', 'Parity']
            except:
                st.session_state.model_feature_names = ['Sex', '30kg ABW', 'Litter size', 'Season', 'Birth weight', 'Parity']
        else:
            st.error(f"Model loading failed: {status}")
            st.stop()

# Page title
st.title("🐷 Growth Rate Prediction Model")
st.markdown("<h3 style='text-align: center;'>Northwest A&F University, Wu.Lab. China</h3>", unsafe_allow_html=True)

# Feature ranges and descriptions
feature_ranges = {
    "Sex": {
        "type": "categorical",
        "options": {"Female": 0, "Male": 1},
        "default": "Female"
    },
    "30kg ABW": {"type": "numerical", "min": 45.0, "max": 100.0, "default": 70.0},
    "Litter size": {"type": "numerical", "min": 0, "max": 35, "default": 15},
    "Season": {
        "type": "categorical",
        "options": {"Spring": 1, "Summer": 2, "Autumn": 3, "Winter": 4},
        "default": "Summer"
    },
    "Birth weight": {"type": "numerical", "min": 0.0, "max": 4.0, "default": 2.0},
    "Parity": {"type": "categorical", "options": [1, 2, 3, 4, 5, 6, 7], "default": 2},
}

# Feature input section
st.header("📊 Enter Feature Values")

# Create input controls in model's expected order
feature_values = []
for feature in st.session_state.model_feature_names:
    properties = feature_ranges[feature]
    
    # Create two columns for better layout
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.write(f"**{feature}:**")
    
    with col2:
        if properties["type"] == "numerical":
            value = st.number_input(
                label="",  # Empty label since we have the feature name in col1
                min_value=float(properties["min"]),
                max_value=float(properties["max"]),
                value=float(properties["default"]),
                key=f"input_{feature}"
            )
        else:
            if isinstance(properties["options"], dict):
                display_options = list(properties["options"].keys())
                selected_label = st.selectbox(
                    label="",
                    options=display_options,
                    index=display_options.index(properties["default"]),
                    key=f"input_{feature}"
                )
                value = properties["options"][selected_label]
            else:
                value = st.selectbox(
                    label="",
                    options=properties["options"],
                    index=properties["options"].index(properties["default"]),
                    key=f"input_{feature}"
                )
    
    feature_values.append(value)
    st.session_state.feature_values[feature] = value

# Create feature DataFrame in correct order
st.session_state.features_df = pd.DataFrame(
    [feature_values], 
    columns=st.session_state.model_feature_names
)

# Prediction section
st.header("🎯 Prediction")

if st.button("🚀 Predict Growth Rate", type="primary", use_container_width=True):
    if not st.session_state.model_loaded:
        st.error("Please load the model first!")
        return
    
    with st.spinner('Calculating prediction...'):
        try:
            # Make prediction
            predicted_value = st.session_state.model.predict(st.session_state.features_df)[0]
            st.session_state.predicted_value = predicted_value
            
            # Display results
            st.success(f"**Predicted Growth Rate: `{predicted_value:.4f}`**")
            
            # Calculate SHAP values for explanation
            try:
                if st.session_state.explainer is None:
                    st.session_state.explainer = shap.TreeExplainer(st.session_state.model)
                
                st.session_state.shap_values = st.session_state.explainer.shap_values(
                    st.session_state.features_df
                )
                
            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {e}")
            
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Explanation section
if st.session_state.predicted_value is not None:
    st.header("🔍 Model Explanation")
    
    # Feature contribution table
    if st.session_state.shap_values is not None:
        st.subheader("Feature Contributions")
        contribution_data = []
        for i, feature in enumerate(st.session_state.model_feature_names):
            shap_value = st.session_state.shap_values[0][i]
            contribution_data.append({
                "Feature": feature,
                "Value": st.session_state.features_df.iloc[0][feature],
                "SHAP Value": shap_value,
                "Impact": "🟢 Increases" if shap_value > 0 else "🔴 Decreases"
            })
        
        contribution_df = pd.DataFrame(contribution_data)
        st.dataframe(contribution_df, use_container_width=True)

    # Download functionality
    st.header("💾 Download Results")
    if st.button("Download Prediction Details", use_container_width=True):
        prediction_details = st.session_state.features_df.copy()
        prediction_details['Predicted_Growth_Rate'] = st.session_state.predicted_value
        
        # Convert to CSV
        csv = prediction_details.to_csv(index=False)
        
        # Provide download button
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="growth_rate_prediction.csv",
            mime="text/csv"
        )

# Sidebar information
st.sidebar.header("About this Model")
st.sidebar.info("""
This is a CatBoost regression model for predicting growth rate in pigs based on various biological features.

**Feature Descriptions:**
- **Sex**: Gender of the pig (Female/Male)
- **30kg ABW**: Age at 30kg body weight (days)
- **Litter size**: Number of piglets in the litter
- **Season**: Birth season of the pig
- **Birth weight**: Individual weight at birth (kg)
- **Parity**: Which litter (1 for first, 2 for second, etc.)
""")

st.sidebar.header("Model Information")
st.sidebar.text("""
Algorithm: CatBoost Regressor
Task: Regression
Target: Growth Rate
""")

# Model status in sidebar
st.sidebar.header("Application Status")
if st.session_state.model_loaded:
    st.sidebar.success("✅ Model loaded and ready")
    st.sidebar.text(f"Features: {len(st.session_state.model_feature_names)}")
else:
    st.sidebar.error("❌ Model not loaded")

# Add some space at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Northwest A&F University • College of Animal Science and Technology</p>", unsafe_allow_html=True)
