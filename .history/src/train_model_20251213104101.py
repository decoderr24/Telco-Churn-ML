import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# === 1. SETUP JALUR FILE OTOMATIS ===
# Mendapatkan lokasi folder tempat script ini berada (folder 'src')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Jalur Dataset: Naik satu level (..) lalu masuk ke 'dataset'
CSV_PATH = os.path.join(BASE_DIR, '../dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Jalur Output Model: Di dalam folder 'src/model'
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'churn_prediction_model.pkl')

# Cek dataset
if not os.path.exists(CSV_PATH):
    print(f"❌ ERROR FATAL: Dataset tidak ditemukan!")
    print(f"Sistem mencari di: {CSV_PATH}")
    print("Pastikan nama folder 'dataset' dan nama file csv benar.")
    exit()

print(f"✅ Dataset ditemukan di: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

# cleaning and preprocessing
print("⚙️ Sedang memproses data...")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

X = df.drop('Churn', axis=1)
y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

num_cols = X.select_dtypes(include=['number']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# Pipeline model
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
    ])

# train model
print("⏳ Sedang melatih model (Random Forest)...")
model_final = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=5, 
        class_weight='balanced', random_state=42
    ))
])

model_final.fit(X, y)

# Buat folder 
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

joblib.dump(model_final, MODEL_PATH)
print(f"🎉 SUKSES! Model berhasil disimpan di: {MODEL_PATH}")