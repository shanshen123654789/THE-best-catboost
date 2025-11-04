import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor

# 设置matplotlib中文字体和格式
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# 加载CatBoost回归模型
try:
    # 尝试使用joblib加载
    model = joblib.load('catboost.pkl')
    st.success("✅ 模型加载成功 (joblib格式)")
except:
    try:
        # 如果joblib失败，尝试使用CatBoost原生格式加载
        model = CatBoostRegressor()
        model.load_model('catboost.cbm')
        st.success("✅ 模型加载成功 (CatBoost原生格式)")
    except Exception as e:
        st.error(f"❌ 模型加载失败: {e}")
        st.stop()

# 特征范围和描述 - 根据你的数据调整
feature_ranges = {
    "30kg ABW": {"type": "numerical", "min": 45.000, "max": 100.000, "default": 70.000},
    "Litter size": {"type": "numerical", "min": 0, "max": 35, "default": 15},
    "Season": {
        "type": "categorical",
        "options": {
            "Spring": 1,
            "Summer": 2,
            "Autumn": 3,
            "Winter": 4
        },
        "default": "Summer"
    },
    "Birth weight": {"type": "numerical", "min": 0.0, "max": 4.0, "default": 2.0},
    "Parity": {"type": "categorical", "options": [1, 2, 3, 4, 5, 6, 7], "default": 2},
    "Sex": {
        "type": "categorical",
        "options": {
            "Female": 0,
            "Male": 1
        },
        "default": "Female"
    },
}

# 页面标题
st.title("Growth Rate Prediction Model with SHAP Visualization")
st.markdown("<h3 style='text-align: center;'>Northwest A&F University, Wu.Lab. China</h3>", unsafe_allow_html=True)

# 输入特征值
st.header("Enter the following feature values:")
feature_values = []
feature_names = list(feature_ranges.keys())

for feature in feature_names:
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
            # For options with labels (Season and Sex)
            display_options = list(properties["options"].keys())
            selected_label = st.selectbox(
                label=f"{feature} (Select a value)",
                options=display_options,
                index=display_options.index(properties["default"]),
                key=feature
            )
            value = properties["options"][selected_label]
        else:
            # For options without labels (Parity)
            value = st.selectbox(
                label=f"{feature} (Select a value)",
                options=properties["options"],
                index=properties["options"].index(properties["default"]),
                key=feature
            )
    feature_values.append(value)

# 创建特征DataFrame
features_df = pd.DataFrame([feature_values], columns=feature_names)

# 初始化session state变量
if 'predicted_value' not in st.session_state:
    st.session_state.predicted_value = None

# 预测按钮
if st.button("Predict Growth Rate"):
    
    # 回归预测
    predicted_value = model.predict(features_df)[0]
    st.session_state.predicted_value = predicted_value
    
    # 显示预测结果
    st.success(f"**Predicted Growth Rate: {predicted_value:.4f}**")
    
    # 创建解释器并计算SHAP值
    st.header("Model Explanation with SHAP")
    
    try:
        # 创建SHAP解释器
        explainer = shap.TreeExplainer(model)
        
        # 计算SHAP值
        shap_values = explainer.shap_values(features_df)
        
        # 创建SHAP摘要图（针对当前样本）
        st.subheader("SHAP Feature Importance for this Prediction")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        # 使用瀑布图展示单个预测的解释
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=features_df.iloc[0],
                feature_names=feature_names
            ),
            show=False
        )
        
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
        
        # 显示特征贡献表格
        st.subheader("Feature Contributions")
        contribution_data = []
        for i, feature in enumerate(feature_names):
            contribution_data.append({
                "Feature": feature,
                "Value": features_df.iloc[0][feature],
                "SHAP Value": shap_values[0][i],
                "Contribution": "Increases prediction" if shap_values[0][i] > 0 else "Decreases prediction"
            })
        
        contribution_df = pd.DataFrame(contribution_data)
        st.dataframe(contribution_df)
        
    except Exception as e:
        st.error(f"SHAP解释生成失败: {e}")
        st.info("请确保已安装shap库: `pip install shap`")

# 添加下载预测结果的功能
st.header("Download Prediction Results")

if st.session_state.predicted_value is not None:
    if st.button("Download Prediction Details"):
        # 创建包含预测详细信息的CSV
        prediction_details = features_df.copy()
        prediction_details['Predicted_Growth_Rate'] = st.session_state.predicted_value
        
        # 转换为CSV
        csv = prediction_details.to_csv(index=False)
        
        # 提供下载按钮
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="growth_rate_prediction.csv",
            mime="text/csv"
        )
else:
    st.info("请先点击 'Predict Growth Rate' 按钮进行预测，然后才能下载结果。")

# 侧边栏信息
st.sidebar.header("About this Model")
st.sidebar.info("""
This is a CatBoost regression model for predicting growth rate based on various pig features.

**Features:**
- 30kg ABW: age at 30 kg body weight 
- Litter size: Number of offspring
- Season: The birth season of the pig
- Birth weight: The pig's individual weight (in kg) recorded at the time of birth.
- Parity
- Sex
""")

st.sidebar.header("Model Performance")
st.sidebar.text("""
Based on test set:
- RMSE: Varies by model
- R²: Varies by model
- MAE: Varies by model
""")