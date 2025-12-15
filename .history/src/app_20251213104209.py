import streamlit as st
import pandas as pd
import joblib

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Telco Churn Prediction", page_icon="📡")

# Judul Aplikasi
st.title("📡 Telco Customer Churn Prediction")
st.write("Aplikasi untuk memprediksi apakah pelanggan akan berhenti berlangganan (Churn) atau setia.")

# Load Model
@st.cache_resource
def load_model():
    try:
        # path model
        model = joblib.load('model/churn_prediction_model.pkl')
        return model
    except FileNotFoundError:
        st.error("File tidak ditemukan. Pastikan file 'churn_prediction_model.pkl' ada di dalam folder 'model/'.")
        return None
model = load_model()

# Form Input Data (Sidebar)
st.sidebar.header("📝 Input Data Pelanggan")

def user_input_features():
    # Kategori Demografi
    st.sidebar.subheader("Demografi")
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    senior_citizen = st.sidebar.selectbox("Senior Citizen (Lansia)", (0, 1), format_func=lambda x: "Ya" if x == 1 else "Tidak")
    partner = st.sidebar.selectbox("Punya Pasangan", ("Yes", "No"))
    dependents = st.sidebar.selectbox("Punya Tanggungan", ("Yes", "No"))
    
    # Kategori Layanan
    st.sidebar.subheader("Layanan")
    tenure = st.sidebar.slider("Lama Berlangganan (Bulan)", 0, 72, 12)
    phone_service = st.sidebar.selectbox("Layanan Telepon", ("Yes", "No"))
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ("No phone service", "No", "Yes"))
    internet_service = st.sidebar.selectbox("Jenis Internet", ("DSL", "Fiber optic", "No"))
    online_security = st.sidebar.selectbox("Keamanan Online", ("No", "Yes", "No internet service"))
    online_backup = st.sidebar.selectbox("Backup Online", ("No", "Yes", "No internet service"))
    device_protection = st.sidebar.selectbox("Proteksi Perangkat", ("No", "Yes", "No internet service"))
    tech_support = st.sidebar.selectbox("Support Teknis", ("No", "Yes", "No internet service"))
    streaming_tv = st.sidebar.selectbox("Streaming TV", ("No", "Yes", "No internet service"))
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ("No", "Yes", "No internet service"))
    
    # Kategori Pembayaran
    st.sidebar.subheader("Pembayaran")
    contract = st.sidebar.selectbox("Kontrak", ("Month-to-month", "One year", "Two year"))
    paperless_billing = st.sidebar.selectbox("Tagihan Tanpa Kertas", ("Yes", "No"))
    payment_method = st.sidebar.selectbox("Metode Bayar", ("Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"))
    monthly_charges = st.sidebar.number_input("Biaya Bulanan ($)", min_value=0.0, value=50.0)
    total_charges = st.sidebar.number_input("Total Biaya ($)", min_value=0.0, value=500.0)

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
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Tampilkan data input
st.subheader('Data Pelanggan:')
st.dataframe(input_df)

# Tombol Prediksi
if st.button('🚀 Prediksi Churn'):
    if model:
        # p
        prediction_proba = model.predict_proba(input_df)[0][1]
        
        # Threshod untuk klasifikasi
        THRESHOLD = 0.6
        
        st.subheader('Hasil Analisis:')
        if prediction_proba > THRESHOLD:
            st.error(f"⚠️ HATI-HATI! Pelanggan berpotensi CHURN.")
            st.write(f"Probabilitas: {prediction_proba:.2%}")
            st.write("**Saran:** Tawarkan diskon atau layanan prioritas segera.")
        else:
            st.success(f"✅ AMAN. Pelanggan diprediksi SETIA.")
            st.write(f"Probabilitas Churn: {prediction_proba:.2%}")