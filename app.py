import streamlit as st
import pandas as pd
import joblib

# Load model pipeline
@st.cache_resource
def load_pipeline():
    pipeline = joblib.load("AdaBoost_best_pipeline.joblib")
    return pipeline

pipeline = load_pipeline()


# tampilan aplikasi
st.set_page_config(page_title="Loan Approval Prediction", layout="centered")

st.title("Loan Approval Prediction App")
st.markdown("""
Aplikasi ini memprediksi apakah pengajuan pinjaman **layak disetujui (Yes)** atau **tidak disetujui (No)**  
berdasarkan data calon peminjam.
""")

st.divider()


# Input data pengguna
st.subheader("Masukkan Data Calon Peminjam")

col1, col2 = st.columns(2)

with col1:
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])

with col2:
    ApplicantIncome = st.number_input("Applicant Income", min_value=0, value=4000)
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0, value=1000)
    LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=0, value=100)
    Loan_Amount_Term = st.number_input("Loan Term (in months)", min_value=0, value=360)
    Credit_History = st.selectbox("Credit History (1 = Good, 0 = Bad)", [1.0, 0.0])
    Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Membuat dataframe dari input
input_data = pd.DataFrame([{
    'Gender': Gender,
    'Married': Married,
    'Dependents': Dependents,
    'Education': Education,
    'Self_Employed': Self_Employed,
    'ApplicantIncome': ApplicantIncome,
    'CoapplicantIncome': CoapplicantIncome,
    'LoanAmount': LoanAmount,
    'Loan_Amount_Term': Loan_Amount_Term,
    'Credit_History': Credit_History,
    'Property_Area': Property_Area
}])


# Prediksi
if st.button("Prediksi"):
    pred = pipeline.predict(input_data)[0]
    prob = pipeline.predict_proba(input_data)[0][1]

    st.subheader("Hasil Prediksi")
    if pred == 1:
        st.success(f"**Disetujui!** (Probabilitas: {prob:.2%})")
    else:
        st.error(f"**Tidak Disetujui.** (Probabilitas: {prob:.2%})")

    st.dataframe(input_data)

st.divider()
