import streamlit as st
import pandas as pd
import pickle

# Load the model and encoders
model_data = pickle.load(open('Customer_churn_model.pkl', 'rb'))
model = model_data["model"]
feature_names = model_data["feature_names"]

encoder = pickle.load(open('encoders.pkl', 'rb'))

# Page configuration
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

st.title("📉 Customer Churn Prediction")
st.markdown("Enter customer details to predict churn likelihood.")

# Two equal columns
col1, _, col2 = st.columns([1, 0.2, 1])

with col1:
    gender = st.selectbox("Gender", ['Male', 'Female'])
    senior = st.selectbox("Senior Citizen", ['Yes', 'No'])
    partner = st.selectbox("Has Partner", ['Yes', 'No'])
    dependents = st.selectbox("Has Dependents", ['Yes', 'No'])
    phone_service = st.selectbox("Phone Service", ['Yes', 'No'])
    multiple_lines = st.selectbox("Multiple Lines", ['Yes', 'No', 'No phone service'])
    internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
    online_backup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
    device_protection = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])

with col2:
    tech_support = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
    streaming_tv = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
    streaming_movies = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])
    contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox("Paperless Billing", ['Yes', 'No'])
    payment_method = st.selectbox("Payment Method", [
        'Electronic check', 'Mailed check', 
        'Bank transfer (automatic)', 'Credit card (automatic)'
    ])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1000.0)

# Prepare input
input_data = {
    'gender': gender,
    'SeniorCitizen': 1 if senior == 'Yes' else 0,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

input_df = pd.DataFrame([input_data])

# Encode categorical fields
for col in input_df.columns:
    if col in encoder:
        input_df[col] = encoder[col].transform(input_df[col])

# Align with training feature order
input_df = input_df[feature_names]

# Predict button
st.markdown("---")
center_col = st.columns(3)[1]
with center_col:
    if st.button("🔍 Predict Churn"):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"⚠️ This customer is likely to churn.\n\n**Probability:** {probability:.2%}")
        else:
            st.success(f"✅ This customer is likely to stay.\n\n**Probability:** {1 - probability:.2%}")
