import streamlit as st
import pandas as pd
import joblib

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Telco Customer Churn Prediction", page_icon="📡")

# 2. Memuat Model
@st.cache_resource
def load_model():
    # Pastikan file .pkl ada di folder yang sama
    return joblib.load('churn_prediction_model.pkl')

try:
    model = load_model()
except FileNotFoundError:
    st.error("File 'churn_prediction_model.pkl' tidak ditemukan. Mohon upload file modelnya.")
    st.stop()

# 3. Judul dan Deskripsi
st.title("📡 Telco Customer Churn Prediction")
st.write("Aplikasi ini memprediksi kemungkinan pelanggan berhenti berlangganan (Churn) berdasarkan data profil dan layanan mereka.")

st.markdown("---")

# 4. Form Input Fitur (Sesuai kolom dataset Anda)
st.sidebar.header("Input Data Pelanggan")

def user_input_features():
    # --- Demografi ---
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    senior_citizen = st.sidebar.selectbox("Senior Citizen (Lansia)", (0, 1), format_func=lambda x: "Ya" if x == 1 else "Tidak")
    partner = st.sidebar.selectbox("Memiliki Pasangan", ("Yes", "No"))
    dependents = st.sidebar.selectbox("Memiliki Tanggungan", ("Yes", "No"))
    
    # --- Layanan ---
    tenure = st.sidebar.slider("Lama Berlangganan (bulan)", 0, 72, 12)
    phone_service = st.sidebar.selectbox("Layanan Telepon", ("Yes", "No"))
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ("No phone service", "No", "Yes"))
    internet_service = st.sidebar.selectbox("Internet Service", ("DSL", "Fiber optic", "No"))
    online_security = st.sidebar.selectbox("Online Security", ("No", "Yes", "No internet service"))
    online_backup = st.sidebar.selectbox("Online Backup", ("No", "Yes", "No internet service"))
    device_protection = st.sidebar.selectbox("Device Protection", ("No", "Yes", "No internet service"))
    tech_support = st.sidebar.selectbox("Tech Support", ("No", "Yes", "No internet service"))
    streaming_tv = st.sidebar.selectbox("Streaming TV", ("No", "Yes", "No internet service"))
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ("No", "Yes", "No internet service"))
    
    # --- Akun ---
    contract = st.sidebar.selectbox("Kontrak", ("Month-to-month", "One year", "Two year"))
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ("Yes", "No"))
    payment_method = st.sidebar.selectbox("Metode Pembayaran", ("Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"))
    monthly_charges = st.sidebar.number_input("Biaya Bulanan", min_value=0.0, value=50.0)
    total_charges = st.sidebar.number_input("Total Biaya", min_value=0.0, value=500.0)

    # Gabungkan menjadi DataFrame
    data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
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
    return pd.DataFrame(data, index=[0])

# Tampilkan data input
input_df = user_input_features()
st.subheader("Data Pelanggan:")
st.dataframe(input_df)

# 5. Tombol Prediksi
if st.button("Prediksi Churn"):
    # Lakukan prediksi
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Hasil Prediksi:")
    
    # Gunakan Threshold Optimal (0.4426) jika ingin hasil lebih sensitif, 
    # atau default (0.5). Di sini saya gunakan 0.5 sesuai standar, 
    # tapi Anda bisa ubah logikanya jika mau.
    if prediction[0] == 1:
        st.error(f"⚠️ Pelanggan Berpotensi CHURN (Probabilitas: {probability:.2%})")
        st.write("**Rekomendasi:** Segera tawarkan promosi retensi pelanggan.")
    else:
        st.success(f"✅ Pelanggan Diprediksi SETIA (Probabilitas Churn: {probability:.2%})")