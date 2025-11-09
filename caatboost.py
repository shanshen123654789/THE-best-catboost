import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
import io
import sys
import os

# 设置matplotlib中文字体和格式
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# 页面配置
st.set_page_config(
    page_title="ADG Prediction Model",
    page_icon="🐷",
    layout="wide"
)

# 初始化session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'predicted_value' not in st.session_state:
    st.session_state.predicted_value = None
if 'shap_values' not in st.session_state:
    st.session_state.shap_values = None
if 'explainer' not in st.session_state:
    st.session_state.explainer = None

# 页面标题
st.title("Average Daily Gain (ADG) Prediction Model with SHAP Visualization")
st.markdown("<h3 style='text-align: center;'>Northwest A&F University, Wu.Lab. China</h3>", unsafe_allow_html=True)

# 加载模型
@st.cache_resource
def load_model():
    """缓存模型加载，避免重复加载"""
    try:
        # 尝试使用joblib加载
        model = joblib.load('catboost.pkl')
        return model, "joblib"
    except Exception as e1:
        try:
            # 如果joblib失败，尝试使用CatBoost原生格式加载
            model = CatBoostRegressor()
            model.load_model('catboost.cbm')
            return model, "CatBoost native"
        except Exception as e2:
            st.error(f"❌ 模型加载失败:")
            st.error(f"Joblib错误: {e1}")
            st.error(f"CatBoost错误: {e2}")
            return None, None

# 侧边栏 - 模型加载
with st.sidebar:
    st.header("Model Configuration")
    
    if st.button("Load Model", type="primary"):
        with st.spinner('Loading model...'):
            model, load_method = load_model()
            if model is not None:
                st.session_state.model = model
                st.session_state.model_loaded = True
                
                # 初始化SHAP解释器（但先不计算）
                try:
                    # 临时重定向stdout避免SHAP冗长输出
                    old_stdout = sys.stdout
                    sys.stdout = open(os.devnull, 'w')
                    
                    st.session_state.explainer = shap.TreeExplainer(model)
                    sys.stdout = old_stdout
                    st.success(f"✅ 模型加载成功 ({load_method}格式)")
                    st.success("✅ SHAP解释器初始化成功")
                except Exception as e:
                    sys.stdout = old_stdout
                    st.warning(f"模型加载成功，但SHAP解释器初始化失败: {e}")
            else:
                st.error("❌ 模型加载失败")

# 如果模型未加载，显示提示
if not st.session_state.model_loaded:
    st.info("👈 请在侧边栏点击'Load Model'按钮来初始化应用")
    st.stop()

# 特征范围和描述
feature_ranges = {
    "30kg ABW": {"type": "numerical", "min": 45.000, "max": 100.000, "default": 70.000},
    "Litter size": {"type": "numerical", "min": 0, "max": 20, "default": 15},
    "Season": {
        "type": "categorical",
        "options": {
            "Spring": 1,
            "Summer": 2,
            "Autumn": 3,
            "Winter": 4
        },
        "default": "Spring"
    },
    "Birth weight (kg)": {"type": "numerical", "min": 0.0, "max": 2.5, "default": 1.5},
    "Parity": {"type": "categorical", "options": [1, 2, 3, 4, 5, 6, 7], "default": 2},
    "Sex": {
        "type": "categorical",
        "options": {
            "Female": 0,
            "Male": 1
        },
        "default": "Male"
    },
}

# 输入特征值
st.header("Enter the following feature values:")
feature_values = []
feature_names = list(feature_ranges.keys())

col1, col2 = st.columns(2)

with col1:
    for feature in feature_names[:3]:
        properties = feature_ranges[feature]
        if properties["type"] == "numerical":
            value = st.number_input(
                label=f"{feature} ({properties['min']} - {properties['max']})",
                min_value=float(properties["min"]),
                max_value=float(properties["max"]),
                value=float(properties["default"]),
                key=feature
            )
        elif properties["type"] == "categorical":
            if isinstance(properties["options"], dict):
                display_options = list(properties["options"].keys())
                selected_label = st.selectbox(
                    label=f"{feature}",
                    options=display_options,
                    index=display_options.index(properties["default"]),
                    key=feature
                )
                value = properties["options"][selected_label]
            else:
                value = st.selectbox(
                    label=f"{feature}",
                    options=properties["options"],
                    index=properties["options"].index(properties["default"]),
                    key=feature
                )
        feature_values.append(value)

with col2:
    for feature in feature_names[3:]:
        properties = feature_ranges[feature]
        if properties["type"] == "numerical":
            value = st.number_input(
                label=f"{feature} ({properties['min']} - {properties['max']})",
                min_value=float(properties["min"]),
                max_value=float(properties["max"]),
                value=float(properties["default"]),
                key=feature
            )
        elif properties["type"] == "categorical":
            if isinstance(properties["options"], dict):
                display_options = list(properties["options"].keys())
                selected_label = st.selectbox(
                    label=f"{feature}",
                    options=display_options,
                    index=display_options.index(properties["default"]),
                    key=feature
                )
                value = properties["options"][selected_label]
            else:
                value = st.selectbox(
                    label=f"{feature}",
                    options=properties["options"],
                    index=properties["options"].index(properties["default"]),
                    key=feature
                )
        feature_values.append(value)

# 创建特征DataFrame
features_df = pd.DataFrame([feature_values], columns=feature_names)

# 预测按钮
if st.button("Predict ADG (g/d)", type="primary"):
    
    with st.spinner('Making prediction and calculating SHAP values...'):
        try:
            # 回归预测
            predicted_value = st.session_state.model.predict(features_df)[0]
            st.session_state.predicted_value = predicted_value
            
            # 计算SHAP值
            if st.session_state.explainer is not None:
                # 临时重定向stdout
                old_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')
                
                shap_values = st.session_state.explainer.shap_values(features_df)
                sys.stdout = old_stdout
                
                st.session_state.shap_values = shap_values
                st.session_state.base_value = st.session_state.explainer.expected_value
            
            # 显示预测结果
            st.success(f"**Predicted ADG: {predicted_value:.2f} g/d**")
            
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# SHAP解释部分
if st.session_state.predicted_value is not None and st.session_state.shap_values is not None:
    st.header("Model Explanation with SHAP")
    
    try:
        # 创建SHAP瀑布图
        st.subheader("SHAP Waterfall Plot")
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 生成SHAP解释
        explanation = shap.Explanation(
            values=st.session_state.shap_values[0],
            base_values=st.session_state.base_value,
            data=features_df.iloc[0],
            feature_names=feature_names
        )
        
        # 绘制瀑布图
        shap.waterfall_plot(explanation, show=False)
        plt.tight_layout()
        
        # 显示图形
        st.pyplot(fig)
        
        # 保存图形到内存用于下载
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='jpg', dpi=600, bbox_inches='tight', facecolor='white')
        img_buffer.seek(0)
        
        # 提供下载按钮
        st.download_button(
            label="Download SHAP Plot (JPG, 600 DPI)",
            data=img_buffer,
            file_name="shap_explanation.jpg",
            mime="image/jpeg"
        )
        
        plt.close(fig)  # 在保存后关闭图形
        
        # 特征贡献表格
        st.subheader("Feature Contributions")
        contribution_data = []
        for i, feature in enumerate(feature_names):
            shap_value = st.session_state.shap_values[0][i]
            contribution_data.append({
                "Feature": feature,
                "Value": features_df.iloc[0][feature],
                "SHAP Value": f"{shap_value:.4f}",
                "Impact": "Increases prediction" if shap_value > 0 else "Decreases prediction",
                "Impact Strength": "Strong" if abs(shap_value) > 0.1 else "Moderate" if abs(shap_value) > 0.01 else "Weak"
            })
        
        contribution_df = pd.DataFrame(contribution_data)
        st.dataframe(contribution_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"SHAP visualization failed: {e}")

# 下载预测结果
st.header("Download Prediction Results")

if st.session_state.predicted_value is not None:
    # 创建包含预测详细信息的CSV
    prediction_details = features_df.copy()
    prediction_details['Predicted_ADG_g_d'] = st.session_state.predicted_value
    
    if st.session_state.shap_values is not None:
        for i, feature in enumerate(feature_names):
            prediction_details[f'SHAP_{feature}'] = st.session_state.shap_values[0][i]
    
    # 转换为CSV
    csv = prediction_details.to_csv(index=False)
    
    # 提供下载按钮
    st.download_button(
        label="Download Prediction Details (CSV)",
        data=csv,
        file_name="ADG_prediction_details.csv",
        mime="text/csv"
    )
else:
    st.info("Please click 'Predict ADG (g/d)' button first to get predictions, then you can download the results.")

# 侧边栏信息
st.sidebar.header("About this Model")
st.sidebar.info("""
This is a CatBoost regression model for predicting Average Daily Gain (ADG) based on various pig features.

**Features:**
- 30kg ABW: Body weight at 30kg
- Litter size: Number of piglets in the litter  
- Season: Birth season
- Birth weight (kg): Individual weight at birth
- Parity: Which litter (1 for first, 2 for second, etc.)
- Sex: Gender of the pig
""")

st.sidebar.header("Model Information")
if st.session_state.model_loaded:
    st.sidebar.write(f"**Model Type:** CatBoost Regressor")
    st.sidebar.write(f"**Features:** {len(feature_names)}")
    st.sidebar.write(f"**SHAP Ready:** {'Yes' if st.session_state.explainer is not None else 'No'}")
