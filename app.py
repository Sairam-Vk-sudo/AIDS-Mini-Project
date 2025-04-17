import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

@st.cache_resource
def load_model():
    df = pd.read_csv("C:/Users/saira/OneDrive/Desktop/AI&DS lab/1000-Records/1000 Records.csv")
    df['Date of Birth'] = pd.to_datetime(df['Date of Birth'], errors='coerce')
    df['Date of Joining'] = pd.to_datetime(df['Date of Joining'], errors='coerce')

    df['Year of Joining'] = df['Date of Joining'].dt.year
    df['Quarter of Joining'] = df['Date of Joining'].dt.quarter
    df['Age in Company (Years)'] = df['Age in Company (Years)'].astype(float)
    df['Salary per Year'] = df['Salary'] / df['Age in Company (Years)']
    df['Last % Hike'] = df['Last % Hike'].str.replace('%', '').astype(float)

    np.random.seed(42)
    df['Attrition'] = 0
    rule1_mask = (df['Age in Company (Years)'] > 5) & (df['Last % Hike'] < 5)
    df.loc[rule1_mask, 'Attrition'] = np.random.choice([0, 1], size=rule1_mask.sum(), p=[0.2, 0.8])
    rule2_mask = (df['Last % Hike'] > 20)
    df.loc[rule2_mask, 'Attrition'] = np.random.choice([0, 1], size=rule2_mask.sum(), p=[0.3, 0.7])
    rule3_mask = (df['Salary'] < df['Salary'].quantile(0.25)) & (df['Age in Company (Years)'] > 3)
    df.loc[rule3_mask, 'Attrition'] = np.random.choice([0, 1], size=rule3_mask.sum(), p=[0.4, 0.6])
    rule4_mask = (df['Year of Joining'] > 2020) & (df['Last % Hike'] > 15)
    df.loc[rule4_mask, 'Attrition'] = np.random.choice([0, 1], size=rule4_mask.sum(), p=[0.5, 0.5])

    df = df.drop_duplicates()

    scaler = MinMaxScaler()
    df[['Salary', 'Last % Hike']] = scaler.fit_transform(df[['Salary', 'Last % Hike']])

    features = ['Age in Company (Years)', 'Salary', 'Last % Hike', 'Year of Joining', 'Quarter of Joining']
    X = df[features]
    y = df['Attrition']

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X, y)
    
    return model, scaler

model, scaler = load_model()

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
app_mode = st.sidebar.radio("Choose Mode", ["Single Prediction", "Batch Prediction via Upload"])

# Single Prediction
if app_mode == "Single Prediction":
    st.title("💼 Employee Attrition Predictor")
    st.subheader("👤 Single Employee Prediction")

    age_in_company = st.slider("Age in Company (Years)", 0, 30, 5)
    salary = st.number_input("Salary", min_value=10000, max_value=500000, value=60000)
    last_hike_percent = st.slider("Last % Hike", 0.0, 50.0, 3.5)
    year_of_joining = st.number_input("Year of Joining", min_value=2000, max_value=2025, value=2021)
    quarter_of_joining = st.selectbox("Quarter of Joining", options=[1, 2, 3, 4])

    if st.button("Predict Attrition"):
        salary_scaled, hike_scaled = scaler.transform([[salary, last_hike_percent]])[0]
        input_df = pd.DataFrame([[age_in_company, salary_scaled, hike_scaled, year_of_joining, quarter_of_joining]],
                                columns=['Age in Company (Years)', 'Salary', 'Last % Hike', 'Year of Joining', 'Quarter of Joining'])
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][pred]

        if pred == 1:
            st.error(f"🚨 Attrition Risk: Likely to Leave (Confidence: {proba * 100:.2f}%)")
        else:
            st.success(f"✅ Attrition Risk: Likely to Stay (Confidence: {proba * 100:.2f}%)")

# Batch Prediction
elif app_mode == "Batch Prediction via Upload":
    st.title("📂 Batch Attrition Predictor")
    st.subheader("📁 Upload File to Predict for Multiple Employees")

    uploaded_file = st.file_uploader("Upload CSV or Excel with required columns", type=["csv", "xlsx"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            user_df = pd.read_csv(uploaded_file)
        else:
            user_df = pd.read_excel(uploaded_file)

        required_cols = ['Age in Company (Years)', 'Salary', 'Last % Hike', 'Year of Joining', 'Quarter of Joining']
        if all(col in user_df.columns for col in required_cols):
            st.success("✅ File loaded successfully!")

            user_df[['Salary', 'Last % Hike']] = scaler.transform(user_df[['Salary', 'Last % Hike']])

            input_data = user_df[required_cols]
            predictions = model.predict(input_data)
            probabilities = model.predict_proba(input_data)

            user_df['Attrition Prediction'] = predictions
            user_df['Confidence (%)'] = [f"{np.max(prob) * 100:.2f}%" for prob in probabilities]

            st.dataframe(user_df[['Age in Company (Years)', 'Salary', 'Last % Hike', 
                                  'Year of Joining', 'Quarter of Joining', 
                                  'Attrition Prediction', 'Confidence (%)']])

            csv = user_df.to_csv(index=False)
            st.download_button("📥 Download Results CSV", csv, "attrition_results.csv", "text/csv")
        else:
            st.error(f"❌ Missing required columns: {required_cols}")
