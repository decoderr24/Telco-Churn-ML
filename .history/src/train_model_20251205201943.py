import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# === 1. LOAD DATASET (Gunakan Path Lengkap Anda) ===
csv_path = r"C:\Users\user\Documents\bengkod\telco-churn-ml\dataset\WA_Fn-UseC_-Telco-Customer-Churn.csv"

if not os.path.exists(csv_path):
    print(f"❌ Error: File tidak ditemukan di: {csv_path}")
    exit()

print("✅ Membaca dataset...")
df = pd.read_csv(csv_path)

# === 2. CLEANING DATA (Sama seperti di Kaggle) ===
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# === 3. PREPROCESSING ===
X = df.drop('Churn', axis=1)
y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

num_cols = X.select_dtypes(include=['number']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# Pipeline Preprocessing
num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

# === 4. DEFINISI MODEL TERBAIK (Tuned Random Forest) ===
model_final = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    ))
])

# === 5. TRAINING MODEL ===
print("⏳ Sedang melatih model ulang di laptop Anda...")
model_final.fit(X, y)

# === 6. SIMPAN MODEL ===
# Simpan ke folder 'model' di dalam 'src'
output_folder = 'model'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

output_path = os.path.join(output_folder, 'churn_prediction_model.pkl')
joblib.dump(model_final, output_path)

print(f"🎉 SUKSES! Model baru yang kompatibel telah disimpan di:\n{output_path}")