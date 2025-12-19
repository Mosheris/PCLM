import streamlit as st
import pandas as pd
import numpy as np
import shap
import json
import joblib
import streamlit.components.v1 as components
import pathlib
from sklearn.ensemble import GradientBoostingClassifier

# ==============================================================================
# 1. 映射字典 (根据您的要求定制)
# ==============================================================================

# Marital Status: 显示文本 -> 编码值
MARITAL_MAP = {
    "Married (including common law)": 0,
    "Single (never married)": 1,
    "Unmarried or Domestic Partner": 1,
    "Separated": 2,
    "Divorced": 2,
    "Widowed": 2
}

# Primary Site
SITE_MAP = {
    "Head of pancreas": 0,
    "Body of pancreas": 1,
    "Tail of pancreas": 2,
    "Others": 3
}

# Histology
HISTOLOGY_MAP = {
    "PanNETs": 0,
    "PDAC": 1,
    "Others": 2
}

# Grade Recode
GRADE_MAP = {
    "Grade I": 0,
    "Grade II": 1,
    "Grade III": 2,
    "Grade IV": 3
}

# T Stage
T_STAGE_MAP = {
    "T1": 0, "T2": 1, "T3": 2, "T4": 3
}

# N Stage
N_STAGE_MAP = {
    "N0": 0, "N1": 1
}

# Binary Options (Surgery, Radiation, Lung.metastasis)
BINARY_MAP = {
    "No": 0, "Yes": 1
}

# ==============================================================================
# 2. 页面配置 & 样式
# ==============================================================================
st.set_page_config(
    page_title="Pancreatic Cancer Liver Metastasis Prediction",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; color: #1f77b4; text-align: center; padding: 1rem 0; font-family: 'Arial'; }
    .prediction-box { padding: 2rem; border-radius: 10px; border-left: 6px solid #1f77b4; margin: 1rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .high-risk { background-color: #ffebee; border-left-color: #e53935; }
    .medium-risk { background-color: #fff8e1; border-left-color: #fb8c00; }
    .low-risk { background-color: #e8f5e9; border-left-color: #43a047; }
    h2 { margin-top: 0; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. 模型加载函数
# ==============================================================================
@st.cache_resource
def load_gbm_model():
    try:
        script_dir = pathlib.Path(__file__).parent
        model_path = script_dir / 'Final_GBM_Model.pkl'

        if not model_path.exists():
            st.error(f"❌ 找不到模型文件: {model_path}")
            st.info("💡 请先运行 训练模型.py 或其他训练脚本进行训练！")
            st.stop()
            
        model = joblib.load(model_path)
        
        # 从模型的feature_names_in_属性获取特征名称
        if hasattr(model, 'feature_names_in_'):
            feature_names = list(model.feature_names_in_)
        else:
            # 备选：使用默认特征名称
            feature_names = [
                "Age", "Tumor.size",
                "Marital.status", "Primary.site", "Histology", 
                "Grade.recode", "T.stage", "N.stage", 
                "Surgery", "Radiation", "Lung.metastasis"
            ]
            
        return model, feature_names
    except Exception as e:
        st.error(f"❌ 模型加载失败: {str(e)}")
        st.stop()

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

model, feature_names = load_gbm_model()
explainer = get_explainer(model)

# ==============================================================================
# 4. 侧边栏：特征输入
# ==============================================================================
st.markdown('<h1 class="main-header">🩺 Pancreatic Cancer Liver Metastasis Risk Prediction System</h1>', unsafe_allow_html=True)
st.markdown("---")

st.sidebar.header("📋 Patient Clinical Feature Input")

with st.sidebar.form("patient_data_form"):
    # 1. 连续变量
    age = st.number_input("Age (years)", min_value=18, max_value=100, value=60, step=1)
    tumor_size = st.number_input("Tumor Size (mm)", min_value=1.0, max_value=200.0, value=30.0, step=1.0)
    
    marital_label = st.selectbox("Marital Status", options=list(MARITAL_MAP.keys()))
    site_label = st.selectbox("Primary Site", options=list(SITE_MAP.keys()))
    histology_label = st.selectbox("Histology", options=list(HISTOLOGY_MAP.keys()))
    grade_label = st.selectbox("Grade", options=list(GRADE_MAP.keys()))
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        t_label = st.selectbox("T Stage", options=list(T_STAGE_MAP.keys()))
    with col_s2:
        n_label = st.selectbox("N Stage", options=list(N_STAGE_MAP.keys()))

    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        surgery_label = st.selectbox("Surgery", options=list(BINARY_MAP.keys()), index=1)
    with col_b2:
        rad_label = st.selectbox("Radiation", options=list(BINARY_MAP.keys()))
    with col_b3:
        lung_label = st.selectbox("Lung Metastasis", options=list(BINARY_MAP.keys()))
    
    submit_button = st.form_submit_button("🔍 Start Prediction", type="primary")

# ==============================================================================
# 5. 预测与展示逻辑
# ==============================================================================
col1, col2 = st.columns([1, 1.2])

if submit_button:
    # --- 数据编码 ---
    input_dict = {
        "Age": age,
        "Tumor.size": tumor_size,
        "Marital.status": MARITAL_MAP[marital_label],
        "Primary.site": SITE_MAP[site_label],
        "Histology": HISTOLOGY_MAP[histology_label],
        "Grade.recode": GRADE_MAP[grade_label],
        "T.stage": T_STAGE_MAP[t_label],
        "N.stage": N_STAGE_MAP[n_label],
        "Surgery": BINARY_MAP[surgery_label],
        "Radiation": BINARY_MAP[rad_label],
        "Lung.metastasis": BINARY_MAP[lung_label]
    }
    
    # 转为 DataFrame 并确保顺序正确
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[feature_names]

    # --- 左侧：预测结果 ---
    with col1:
        st.subheader("📊 Risk Prediction Results")
        try:
            # 预测概率
            prob_array = model.predict_proba(input_df)
            risk_prob = prob_array[0][1] # 取正类概率
            risk_percent = risk_prob * 100

            # 风险定级逻辑 (根据常规临床阈值设定，可调整)
            if risk_prob < 0.20:
                level, css, emoji = "Low Risk", "low-risk", "🟢"
                advice = "The patient has a relatively low probability of liver metastasis. Routine follow-up according to standard guidelines is recommended."
            elif risk_prob < 0.50:
                level, css, emoji = "Medium Risk", "medium-risk", "🟡"
                advice = "The patient has a moderate risk of liver metastasis. Shorter follow-up intervals and close monitoring of liver imaging are recommended."
            else:
                level, css, emoji = "High Risk", "high-risk", "🔴"
                advice = "The patient has a very high probability of liver metastasis! Immediate enhanced CT/MRI screening and consideration of adjuvant therapy are recommended."

            st.markdown(f"""
            <div class="prediction-box {css}">
                <h2 style="color: black;">{emoji} Liver Metastasis Probability: {risk_percent:.1f}%</h2>
                <h3 style="color: black;">Risk Level: {level}</h3>
                <p style="margin-top:10px; font-size:1.05rem; color: black;">{advice}</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("View Input Data"):
                st.dataframe(input_df, hide_index=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")

    # --- Right side: SHAP explanation ---
    with col2:
        st.subheader("🔥 Feature Attribution Analysis (SHAP)")
        try:
            with st.spinner("Calculating feature contributions..."):
                # 计算 SHAP
                shap_values = explainer.shap_values(input_df)
                
                # 处理SHAP值 - 对于二分类模型
                if isinstance(shap_values, list):
                    # 列表形式，通常有两个元素（两个类别）
                    if len(shap_values) > 1:
                        shap_target = shap_values[1][0]  # 取正类（第二个类别）的第一个样本
                    else:
                        shap_target = shap_values[0][0]
                elif isinstance(shap_values, np.ndarray):
                    if len(shap_values.shape) == 3:
                        # 三维数组 (samples, features, classes)
                        shap_target = shap_values[0, :, 1] if shap_values.shape[2] > 1 else shap_values[0, :, 0]
                    elif len(shap_values.shape) == 2:
                        # 二维数组 (samples, features)
                        shap_target = shap_values[0, :]
                    else:
                        shap_target = shap_values[0]
                else:
                    shap_target = shap_values[0] if hasattr(shap_values, '__getitem__') else shap_values

                # 获取基准值
                expected_val = explainer.expected_value
                if isinstance(expected_val, (np.ndarray, list)):
                    if len(expected_val) > 1:
                        base_val = expected_val[1]  # 取正类的基准值
                    else:
                        base_val = expected_val[0]
                else:
                    base_val = expected_val

                # 绘制 Force Plot
                try:
                    shap_html = shap.force_plot(
                        base_val,
                        shap_target,
                        input_df.iloc[0],
                        feature_names=feature_names,
                        matplotlib=False
                    )
                    components.html(shap.getjs() + shap_html.html(), height=350, scrolling=True)
                except Exception as plot_error:
                    st.warning(f"Force plot rendering failed, showing alternative visualization...")
                    # 备选方案：用条形图显示SHAP值
                    shap_df = pd.DataFrame({
                        'Feature': feature_names,
                        'SHAP Value': shap_target
                    }).sort_values('SHAP Value', key=abs, ascending=False).head(10)
                    
                    col_chart1, col_chart2 = st.columns(2)
                    with col_chart1:
                        st.bar_chart(shap_df.set_index('Feature')['SHAP Value'])

                st.info("""
                **Chart Interpretation:**
                - 🟥 **Red bars**: Factors pushing risk **upward** (e.g., T4 stage, no surgery).
                - 🟦 **Blue bars**: Factors pushing risk **downward** (e.g., older age, smaller tumor).
                - The length of the bar represents the strength of influence.
                """)

        except Exception as e:
            st.error(f"SHAP analysis failed: {e}")

else:
    col1.info("👈 Please enter patient information in the left sidebar and click the 'Start Prediction' button.")
    col2.empty()

# 页脚
st.markdown("---")
st.markdown("<div style='text-align: center; color: #888;'>Powered by Sklearn GBM & Streamlit | Clinical Decision Support System</div>", unsafe_allow_html=True)