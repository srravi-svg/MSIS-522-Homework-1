"""
COVID-19 Mortality Risk Prediction — Streamlit App
====================================================
MSIS 522 Homework 1
Tabs: Executive Summary | Descriptive Analytics | Model Performance | Explainability & Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import shap
import os

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(page_title="COVID-19 Mortality Prediction", layout="wide", page_icon="🏥")
st.title("🏥 COVID-19 Mortality Risk Prediction")
st.caption("MSIS 522 — The Complete Data Science Workflow")

# ──────────────────────────────────────────────
# Load assets
# ──────────────────────────────────────────────

@st.cache_data
def load_data():
    return pd.read_csv("covid_balanced.csv")

@st.cache_resource
def load_models():
    models = {}
    models['Logistic Regression'] = joblib.load("models/logistic_regression.pkl")
    models['Decision Tree']       = joblib.load("models/decision_tree.pkl")
    models['Random Forest']       = joblib.load("models/random_forest.pkl")
    models['LightGBM']            = joblib.load("models/lightgbm.pkl")
    # Neural Network: try loading Keras model; skip if TensorFlow unavailable
    # (TensorFlow is too heavy for Streamlit Cloud's free tier)
    try:
        from tensorflow import keras
        models['Neural Network'] = keras.models.load_model("models/neural_network.keras")
    except Exception:
        pass  # NN metrics still shown in comparison table from saved CSV
    return models

@st.cache_resource
def load_scaler():
    return joblib.load("models/scaler.pkl")

@st.cache_data
def load_comparison():
    return pd.read_csv("models/model_comparison.csv", index_col=0)

@st.cache_data
def load_best_params():
    return joblib.load("models/best_params_all.pkl")

@st.cache_data
def load_nn_history():
    with open("models/nn_history.json") as f:
        return json.load(f)

@st.cache_data
def load_shap_artifacts():
    vals = np.load("models/shap_values.npy")
    ev   = joblib.load("models/shap_expected_value.pkl")
    return vals, ev

df             = load_data()
models         = load_models()
scaler         = load_scaler()
comparison_df  = load_comparison()
best_params    = load_best_params()
feature_names  = joblib.load("models/feature_names.pkl")

# ──────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Executive Summary",
    "📊 Descriptive Analytics",
    "🤖 Model Performance",
    "🔍 Explainability & Prediction"
])

# ══════════════════════════════════════════════
# TAB 1 — Executive Summary
# ══════════════════════════════════════════════
with tab1:
    st.header("Executive Summary")

    st.subheader("Dataset & Prediction Task")
    st.write("""
    This project analyzes patient-level data from Mexico's national COVID-19 epidemiological
    surveillance system. The dataset captures **16 features** for each patient, including
    demographics (age, sex, pregnancy status), whether they were hospitalized, their COVID-19
    test result, and ten pre-existing conditions — diabetes, COPD, asthma, hypertension,
    obesity, cardiovascular disease, chronic renal disease, immunosuppression, tobacco use,
    and other diseases.

    The **prediction target** is `DEATH`, a binary variable indicating whether a patient
    ultimately died (1) or survived (0). From the original ~1 million patient records, we
    constructed a balanced sample of **10,000 patients** (5,000 who died, 5,000 who survived)
    to ensure our models receive equal exposure to both outcomes during training.
    """)

    st.subheader("Why This Problem Matters")
    st.write("""
    During pandemic surges, hospitals face agonizing triage decisions: which patients should
    receive scarce ICU beds, ventilators, and aggressive treatment? A reliable mortality risk
    model — built from easily obtainable patient attributes like age, pre-existing conditions,
    and hospitalization status — could help clinicians **identify high-risk patients earlier**
    and allocate resources more effectively. Even small improvements in early risk
    stratification can translate to lives saved, reduced hospital strain, and more equitable
    care. Beyond COVID-19, the methodology demonstrated here generalizes to any disease where
    early risk scoring from routine clinical data can inform treatment decisions.
    """)

    st.subheader("Approach & Key Findings")
    st.write("""
    We trained and compared five models of increasing complexity: **Logistic Regression**
    (baseline), **Decision Tree**, **Random Forest**, **LightGBM** (gradient boosted trees),
    and a **Neural Network** (two-hidden-layer MLP). All tree-based models were tuned via
    5-fold cross-validation with GridSearchCV, using F1 score as the optimization metric.

    **LightGBM** emerged as the top performer across nearly all metrics (F1, AUC-ROC, and
    overall accuracy), closely followed by Random Forest. The Neural Network performed
    competitively but did not surpass the gradient-boosted ensemble — a well-known pattern
    for structured tabular data. The simple Decision Tree, while less accurate, provides
    full transparency and may be preferable in settings where every prediction must be
    human-interpretable.

    SHAP analysis on the LightGBM model revealed that **hospitalization status**, **age**,
    and **pneumonia** are the three most influential predictors of mortality — aligning with
    established clinical knowledge. These insights are directly actionable: a triage protocol
    that flags hospitalized patients over age 60 with pneumonia would capture the highest-risk
    group identified by the model.
    """)

    st.divider()
    st.write("Navigate to the other tabs for detailed visualizations, model metrics, and interactive predictions.")


# ══════════════════════════════════════════════
# TAB 2 — Descriptive Analytics
# ══════════════════════════════════════════════
with tab2:
    st.header("Descriptive Analytics")

    # ---- Target Distribution ----
    st.subheader("Target Variable Distribution")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    counts = df['DEATH'].value_counts().sort_index()
    bars = ax.bar(['Survived (0)', 'Died (1)'], counts.values,
                  color=['#2ecc71', '#e74c3c'], edgecolor='black')
    for bar, c in zip(bars, counts.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+40,
                str(c), ha='center', fontweight='bold')
    ax.set_ylabel('Count'); ax.set_title('Death Distribution')
    st.pyplot(fig)
    st.write("""
    The dataset contains exactly **5,000 patients who survived and 5,000 who died**, 
    created through deliberate balanced sampling from the original ~1M-row dataset. This 
    50/50 balance ensures our models are not biased toward the majority class and makes 
    accuracy a meaningful metric alongside F1, precision, and recall.
    """)

    st.divider()

    # ---- Age Distribution ----
    st.subheader("Age Distribution by Outcome")
    fig, ax = plt.subplots(figsize=(7, 4))
    for val, color, label in [(0,'#2ecc71','Survived'),(1,'#e74c3c','Died')]:
        ax.hist(df[df['DEATH']==val]['AGE'], bins=40, alpha=0.55,
                color=color, label=label, edgecolor='white')
    ax.set_xlabel('Age'); ax.set_ylabel('Count'); ax.legend()
    ax.set_title('Age Distribution by Survival Outcome')
    st.pyplot(fig)
    st.write("""
    Age is the most visually striking differentiator between outcomes. Patients who died are 
    concentrated in the 55–80 age range, while survivors span a much broader and younger 
    distribution. This confirms that advanced age is a primary COVID-19 mortality risk factor 
    and will likely be a dominant feature in our predictive models.
    """)

    st.divider()

    # ---- Boxplot ----
    st.subheader("Age Boxplot: Survived vs Died")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.boxplot(data=df, x='DEATH', y='AGE', palette={0:'#2ecc71',1:'#e74c3c'}, ax=ax)
    ax.set_xticklabels(['Survived','Died']); ax.set_title('Age by Outcome')
    st.pyplot(fig)
    st.write(f"""
    The median age of deceased patients ({df[df['DEATH']==1]['AGE'].median():.0f}) is 
    substantially higher than survivors ({df[df['DEATH']==0]['AGE'].median():.0f}). The 
    interquartile range for the deceased group sits almost entirely above that of survivors, 
    reinforcing age as a powerful standalone predictor.
    """)

    st.divider()

    # ---- Mortality by Comorbidity ----
    st.subheader("Mortality Rate by Pre-existing Condition")
    comorbidities = ['PNEUMONIA','DIABETES','COPD','ASTHMA','IMMUNOSUPPRESSION',
                     'HYPERTENSION','CARDIOVASCULAR','RENAL_CHRONIC','OBESITY','TOBACCO']
    mort = []
    for c in comorbidities:
        mort.append({'Comorbidity': c.replace('_',' ').title(),
                     'With': df[df[c]==1]['DEATH'].mean()*100,
                     'Without': df[df[c]==0]['DEATH'].mean()*100})
    mr = pd.DataFrame(mort)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(mr)); w = 0.35
    ax.bar(x-w/2, mr['With'], w, label='With Condition', color='#e74c3c', edgecolor='black')
    ax.bar(x+w/2, mr['Without'], w, label='Without Condition', color='#2ecc71', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(mr['Comorbidity'], rotation=45, ha='right')
    ax.set_ylabel('Mortality Rate (%)'); ax.legend(); ax.set_title('Mortality Rate by Comorbidity')
    st.pyplot(fig)
    st.write("""
    Pneumonia stands out with the largest gap in mortality rates — patients with pneumonia 
    die at dramatically higher rates. COPD, chronic renal disease, and cardiovascular conditions 
    also show meaningfully elevated mortality. Interestingly, asthma and obesity show smaller 
    differences, suggesting they contribute less to mortality risk independently.
    """)

    st.divider()

    # ---- Interaction Plot ----
    st.subheader("Hospitalization × Pneumonia Interaction")
    ct = df.groupby(['HOSPITALIZED','PNEUMONIA'])['DEATH'].mean().reset_index()
    ct['Group'] = ct.apply(lambda r: f"Hosp={int(r.HOSPITALIZED)}, Pneu={int(r.PNEUMONIA)}", axis=1)
    fig, ax = plt.subplots(figsize=(6, 4))
    colors_int = ['#2ecc71','#f39c12','#e67e22','#e74c3c']
    bars = ax.bar(ct['Group'], ct['DEATH']*100, color=colors_int, edgecolor='black')
    for bar, val in zip(bars, ct['DEATH']*100):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f'{val:.1f}%', ha='center', fontweight='bold')
    ax.set_ylabel('Mortality Rate (%)'); ax.set_title('Hospitalization × Pneumonia')
    st.pyplot(fig)
    st.write("""
    This reveals a powerful interaction: patients who are both hospitalized AND have pneumonia 
    face the highest mortality, while non-hospitalized patients without pneumonia face the lowest. 
    The combination is far more predictive than either factor alone, suggesting these features 
    interact multiplicatively.
    """)

    st.divider()

    # ---- Correlation Heatmap ----
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True, linewidths=.5, ax=ax,
                annot_kws={"size": 7})
    ax.set_title('Feature Correlation Heatmap')
    st.pyplot(fig)
    st.write("""
    HOSPITALIZED and PNEUMONIA show the strongest positive correlations with DEATH. AGE, 
    DIABETES, and HYPERTENSION also show meaningful correlations. Notable inter-feature 
    correlations exist between DIABETES and HYPERTENSION (common comorbidity pairing) and 
    between HOSPITALIZED and PNEUMONIA, which may introduce some multicollinearity for linear 
    models but is handled naturally by tree-based approaches.
    """)


# ══════════════════════════════════════════════
# TAB 3 — Model Performance
# ══════════════════════════════════════════════
with tab3:
    st.header("Model Performance")

    # ---- Comparison Table ----
    st.subheader("Model Comparison Table")
    st.dataframe(comparison_df.style.highlight_max(axis=0, color='#d4edda')
                 .format("{:.4f}"), use_container_width=True)

    # ---- Best Hyperparameters ----
    st.subheader("Best Hyperparameters")
    for model_name, params in best_params.items():
        st.write(f"**{model_name}:** `{params}`")

    st.divider()

    # ---- Bar Chart ----
    st.subheader("F1 & AUC-ROC Comparison")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ['#1abc9c','#3498db','#e74c3c','#9b59b6','#f39c12']
    model_names = comparison_df.index.tolist()

    for i, (metric, title) in enumerate([('F1','F1 Score'),('AUC-ROC','AUC-ROC')]):
        bars = axes[i].bar(model_names, comparison_df[metric],
                           color=colors[:len(model_names)], edgecolor='black')
        for bar, val in zip(bars, comparison_df[metric]):
            axes[i].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                         f'{val:.4f}', ha='center', fontsize=8, fontweight='bold')
        axes[i].set_title(title, fontweight='bold')
        axes[i].set_ylim(0, 1.08)
        axes[i].tick_params(axis='x', rotation=25)
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()

    # ---- ROC Curves ----
    st.subheader("ROC Curves — All Models")
    if os.path.exists("plots/roc_all_models.png"):
        st.image("plots/roc_all_models.png")
    else:
        st.info("ROC curve image not found. Run the notebook first to generate plots.")

    st.divider()

    # ---- Individual model plots ----
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Decision Tree Visualization")
        if os.path.exists("plots/decision_tree_viz.png"):
            st.image("plots/decision_tree_viz.png")
        st.subheader("Decision Tree CV Heatmap")
        if os.path.exists("plots/dt_gridsearch_heatmap.png"):
            st.image("plots/dt_gridsearch_heatmap.png")
    with col2:
        st.subheader("Random Forest CV Heatmap")
        if os.path.exists("plots/rf_gridsearch_heatmap.png"):
            st.image("plots/rf_gridsearch_heatmap.png")
        st.subheader("Threshold Analysis (LightGBM)")
        if os.path.exists("plots/threshold_analysis.png"):
            st.image("plots/threshold_analysis.png")

    st.divider()

    # ---- Neural Network Training ----
    st.subheader("Neural Network Training History")
    try:
        hist = load_nn_history()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(hist['loss'], label='Train'); axes[0].plot(hist['val_loss'], label='Val')
        axes[0].set_title('Loss'); axes[0].legend(); axes[0].set_xlabel('Epoch')
        axes[1].plot(hist['accuracy'], label='Train'); axes[1].plot(hist['val_accuracy'], label='Val')
        axes[1].set_title('Accuracy'); axes[1].legend(); axes[1].set_xlabel('Epoch')
        plt.tight_layout()
        st.pyplot(fig)
    except Exception:
        if os.path.exists("plots/nn_training_history.png"):
            st.image("plots/nn_training_history.png")


# ══════════════════════════════════════════════
# TAB 4 — Explainability & Interactive Prediction
# ══════════════════════════════════════════════
with tab4:
    st.header("Explainability & Interactive Prediction")

    # ---- SHAP Plots ----
    st.subheader("SHAP Analysis (LightGBM)")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.write("**Summary Plot (Beeswarm)**")
        if os.path.exists("plots/shap_summary.png"):
            st.image("plots/shap_summary.png")
        else:
            st.info("Run notebook to generate SHAP plots.")
    with col_s2:
        st.write("**Feature Importance (Bar)**")
        if os.path.exists("plots/shap_bar.png"):
            st.image("plots/shap_bar.png")
        else:
            st.info("Run notebook to generate SHAP plots.")

    st.divider()

    # ── Interactive Prediction ──
    st.subheader("🔮 Interactive Prediction")
    st.write("Adjust the patient features below and see the predicted mortality risk in real time.")

    # Model selector
    model_choice = st.selectbox("Select model for prediction:",
                                list(models.keys()))

    # Input controls — organized in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 0, 105, 55)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
        hospitalized = st.selectbox("Hospitalized", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        pneumonia = st.selectbox("Pneumonia", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        covid_positive = st.selectbox("COVID Positive", [0, 1], index=1,
                                       format_func=lambda x: "No" if x == 0 else "Yes")

    with col2:
        diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        obesity = st.selectbox("Obesity", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        copd = st.selectbox("COPD", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        asthma = st.selectbox("Asthma", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    with col3:
        immunosuppression = st.selectbox("Immunosuppression", [0, 1],
                                          format_func=lambda x: "No" if x == 0 else "Yes")
        cardiovascular = st.selectbox("Cardiovascular Disease", [0, 1],
                                       format_func=lambda x: "No" if x == 0 else "Yes")
        renal_chronic = st.selectbox("Chronic Renal Disease", [0, 1],
                                      format_func=lambda x: "No" if x == 0 else "Yes")
        tobacco = st.selectbox("Tobacco Use", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        pregnant = st.selectbox("Pregnant", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        other_disease = st.selectbox("Other Disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    # Build input vector in the EXACT column order the models expect
    input_dict = {
        'SEX': sex, 'HOSPITALIZED': hospitalized, 'PNEUMONIA': pneumonia,
        'AGE': age, 'PREGNANT': pregnant, 'DIABETES': diabetes,
        'COPD': copd, 'ASTHMA': asthma, 'IMMUNOSUPPRESSION': immunosuppression,
        'HYPERTENSION': hypertension, 'OTHER_DISEASE': other_disease,
        'CARDIOVASCULAR': cardiovascular, 'OBESITY': obesity,
        'RENAL_CHRONIC': renal_chronic, 'TOBACCO': tobacco,
        'COVID_POSITIVE': covid_positive
    }
    input_df = pd.DataFrame([input_dict])[feature_names]  # enforce column order

    # Predict
    if st.button("🔍 Predict", type="primary"):
        chosen_model = models[model_choice]

        # Models that need scaling
        if model_choice in ['Logistic Regression', 'Neural Network']:
            input_arr = scaler.transform(input_df)
        else:
            input_arr = input_df

        if model_choice == 'Neural Network':
            proba = float(chosen_model.predict(input_arr).ravel()[0])
            prediction = int(proba > 0.5)
        else:
            prediction = int(chosen_model.predict(input_arr)[0])
            proba = float(chosen_model.predict_proba(input_arr)[0][1])

        # Display results
        st.divider()
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            if prediction == 1:
                st.error(f"⚠️ **Prediction: HIGH RISK (Death)**")
            else:
                st.success(f"✅ **Prediction: LOW RISK (Survival)**")
        with res_col2:
            st.metric("Mortality Probability", f"{proba:.1%}")
        with res_col3:
            st.metric("Survival Probability", f"{1 - proba:.1%}")

        # ---- SHAP Waterfall for this input ----
        st.divider()
        st.subheader("SHAP Waterfall — Why This Prediction?")
        try:
            lgb_model = models.get('LightGBM')
            if lgb_model is not None:
                explainer_live = shap.TreeExplainer(lgb_model)
                sv = explainer_live.shap_values(input_df)
                if isinstance(sv, list):
                    sv = sv[1]
                ev = explainer_live.expected_value
                if isinstance(ev, list):
                    ev = ev[1]

                exp = shap.Explanation(
                    values=sv[0],
                    base_values=float(ev),
                    data=input_df.iloc[0].values,
                    feature_names=list(input_df.columns)
                )
                fig_wf, ax_wf = plt.subplots(figsize=(10, 6))
                shap.plots.waterfall(exp, show=False)
                plt.tight_layout()
                st.pyplot(fig_wf)

                st.caption("""
                This waterfall plot shows how each feature pushes the prediction 
                from the baseline (average) toward the final output. Red bars push 
                toward higher mortality risk; blue bars push toward survival.
                """)
            else:
                st.info("LightGBM model not loaded — waterfall plot unavailable.")
        except Exception as e:
            st.warning(f"Could not generate SHAP waterfall: {e}")
