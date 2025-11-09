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
if 'model_feature_names' not in st.session_state:
    st.session_state.model_feature_names = None

# 页面标题（居中显示）
st.markdown("<h1 style='text-align: center; font-size: 36px;'>Average Daily Gain (ADG) Prediction Model with SHAP Visualization</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; font-size: 24px;'>Northwest A&F University, Wu.Lab. China</h3>", unsafe_allow_html=True)

# 注入CSS来修改字体和字号
st.markdown("""
    <style>
    /* 基础样式 */
    .stTextInput, .stNumberInput, .stSelectbox, .stTextArea, .stRadio, .stSlider {
        font-family: 'Times New Roman', serif;
        font-size: 20px;
    }
    .stButton>button {
        font-family: 'Times New Roman', serif;
        font-size: 20px;
        padding: 12px 24px;
    }

    /* 增大所有标签字体大小 */
    .stNumberInput label, .stSelectbox label, .stTextInput label, .stRadio label {
        font-size: 24px !important;
        font-family: 'Times New Roman', serif !important;
        font-weight: bold !important;
    }
    
    /* 特定输入框标签的字体更大 */
    .stNumberInput label, .stSelectbox label {
        font-size: 26px !important;
        font-weight: bold !important;
    }

    /* 使输入框整体大小变大 */
    .stNumberInput input, .stTextInput input, .stSelectbox select {
        font-size: 22px;
        padding: 12px;
        height: 50px;
    }
    
    /* 预测结果样式 - 非常大的字体 */
    .big-result {
        font-size: 32px !important;
        font-weight: bold !important;
        color: #1f77b4 !important;
        text-align: center !important;
    }
    
    /* 章节标题样式 */
    .section-header {
        font-size: 28px !important;
        font-weight: bold !important;
        color: #2e86ab !important;
        margin-bottom: 20px !important;
    }
    
    /* 成功消息样式 */
    .stAlert {
        font-size: 24px !important;
    }
    
    /* 侧边栏样式 */
    .css-1d391kg {
        font-size: 18px !important;
    }
    </style>
""", unsafe_allow_html=True)

# 加载模型并获取特征顺序
@st.cache_resource
def load_model():
    """缓存模型加载，避免重复加载"""
    try:
        # 尝试使用joblib加载
        model = joblib.load('catboost.pkl')
        
        # 获取模型的特征名称（训练时的顺序）
        if hasattr(model, 'feature_names_'):
            feature_names = model.feature_names_
        else:
            # 如果没有特征名称属性，使用默认顺序
            feature_names = ['30kg ABW', 'Litter size', 'Season', 'Birth weight', 'Parity', 'Sex']
        
        return model, "joblib", feature_names
        
    except Exception as e1:
        try:
            # 如果joblib失败，尝试使用CatBoost原生格式加载
            model = CatBoostRegressor()
            model.load_model('catboost.cbm')
            
            # 获取特征名称
            if hasattr(model, 'feature_names_'):
                feature_names = model.feature_names_
            else:
                feature_names = ['30kg ABW', 'Litter size', 'Season', 'Birth weight', 'Parity', 'Sex']
                
            return model, "CatBoost native", feature_names
        except Exception as e2:
            st.error(f"❌ 模型加载失败: {e1}, {e2}")
            return None, None, None

# 直接加载模型
with st.spinner('Loading model...'):
    model, load_method, feature_names = load_model()
    if model is not None:
        st.session_state.model = model
        st.session_state.model_loaded = True
        st.session_state.model_feature_names = feature_names
        
        # 初始化SHAP解释器
        try:
            st.session_state.explainer = shap.TreeExplainer(model)
        except Exception as e:
            st.warning(f"模型加载成功，但SHAP解释器初始化失败: {e}")
    else:
        st.error("❌ 模型加载失败")

# 如果模型未加载，显示提示
if not st.session_state.model_loaded:
    st.info("👈 请确保模型已成功加载")
    st.stop()

# 特征范围和描述 - 注意：这里使用模型的特征名称顺序
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
    "Birth weight": {"type": "numerical", "min": 0.0, "max": 2.5, "default": 1.5},
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

# 按照模型的特征顺序重新排列特征
ordered_feature_names = st.session_state.model_feature_names

# 输入特征值 - 使用自定义CSS类来放大字体
st.markdown('<div class="section-header">Enter the following feature values:</div>', unsafe_allow_html=True)
feature_values_dict = {}

# 使用两列布局来展示输入框
col1, col2 = st.columns([1, 1])

# 第一列特征
with col1:
    for i, feature in enumerate(ordered_feature_names[:3]):
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
        feature_values_dict[feature] = value

# 第二列特征
with col2:
    for i, feature in enumerate(ordered_feature_names[3:], 3):
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
        feature_values_dict[feature] = value

# 创建特征DataFrame - 按照模型的特征顺序
feature_values_ordered = [feature_values_dict[name] for name in ordered_feature_names]
features_df = pd.DataFrame([feature_values_ordered], columns=ordered_feature_names)

# 预测按钮 - 使用更大的按钮
if st.button("Predict ADG (g/d)", type="primary", use_container_width=True):
    
    with st.spinner('Making prediction and calculating SHAP values...'):
        try:
            # 回归预测
            predicted_value = st.session_state.model.predict(features_df)[0]
            st.session_state.predicted_value = predicted_value
            
            # 计算SHAP值
            if st.session_state.explainer is not None:
                shap_values = st.session_state.explainer.shap_values(features_df)
                st.session_state.shap_values = shap_values
                st.session_state.base_value = st.session_state.explainer.expected_value
            
            # 显示预测结果 - 使用大字体
            st.markdown(f'<div class="big-result">Predicted ADG: {predicted_value:.2f} g/d</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            # 显示调试信息
            st.error(f"输入特征顺序: {list(features_df.columns)}")
            st.error(f"模型期望顺序: {ordered_feature_names}")

# SHAP解释部分
if st.session_state.predicted_value is not None and st.session_state.shap_values is not None:
    st.markdown('<div class="section-header">Model Explanation with SHAP</div>', unsafe_allow_html=True)
    
    try:
        # 创建SHAP瀑布图
        st.markdown('<div style="font-size: 24px; font-weight: bold; margin-bottom: 15px;">SHAP Waterfall Plot</div>', unsafe_allow_html=True)
        
        # 生成SHAP解释
        explanation = shap.Explanation(
            values=st.session_state.shap_values[0],
            base_values=st.session_state.base_value,
            data=features_df.iloc[0],
            feature_names=ordered_feature_names
        )
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 绘制瀑布图
        shap.waterfall_plot(explanation, show=False)
        plt.tight_layout()
        
        # 显示图形
        st.pyplot(fig)
        
        # 保存图形到内存用于下载（JPG格式，DPI=600）
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='jpg', dpi=600, bbox_inches='tight', facecolor='white')
        img_buffer.seek(0)
        
        # 提供下载JPG按钮
        st.download_button(
            label="Download SHAP Plot (JPG, 600 DPI)",
            data=img_buffer,
            file_name="shap_explanation.jpg",
            mime="image/jpeg"
        )
        
        # 保存图形到内存用于下载（PDF格式，DPI=1200）
        pdf_buffer = io.BytesIO()
        fig.savefig(pdf_buffer, format='pdf', dpi=1200, bbox_inches='tight', facecolor='white')
        pdf_buffer.seek(0)
        
        # 提供下载PDF按钮
        st.download_button(
            label="Download SHAP Plot (PDF, 1200 DPI)",
            data=pdf_buffer,
            file_name="shap_explanation.pdf",
            mime="application/pdf"
        )
        
        plt.close(fig)  # 在保存后关闭图形
        
    except Exception as e:
        st.error(f"SHAP waterfall plot failed: {e}")
        st.info("尝试替代的SHAP可视化...")

# 下载预测结果
st.markdown('<div class="section-header">Download Prediction Results</div>', unsafe_allow_html=True)

if st.session_state.predicted_value is not None:
    # 创建包含预测详细信息的CSV
    prediction_details = features_df.copy()
    prediction_details['Predicted_ADG_g_d'] = st.session_state.predicted_value
    
    if st.session_state.shap_values is not None:
        for i, feature in enumerate(ordered_feature_names):
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
This is a CatBoost regression model for predicting Average Daily Gain (ADG) in pigs based on various biological features.

**Feature Descriptions:**
- **30kg ABW**: Age at 30kg body weight
- **Litter size**: Number of piglets in the litter
- **Season**: Birth season of the pig
- **Birth weight**: Individual weight at birth
- **Parity**: Which litter (1 for first, 2 for second, etc.)
- **Sex**: Gender of the pig
""")

st.sidebar.header("Model Information")
st.sidebar.text("""
Algorithm: CatBoost Regressor
Task: Regression
Target: Average Daily Gain (ADG)
""")

# 模型状态在侧边栏
st.sidebar.header("Application Status")
if st.session_state.model_loaded:
    st.sidebar.success("✅ Model loaded and ready")
    st.sidebar.text(f"Features: {len(st.session_state.model_feature_names)}")
else:
    st.sidebar.error("❌ Model not loaded")

# 底部信息
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray; font-size: 18px;'>Northwest A&F University • College of Animal Science and Technology</p>", unsafe_allow_html=True)
