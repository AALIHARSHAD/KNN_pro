import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Heart Disease Prediction - KNN", layout="centered")

st.title("❤️ Heart Disease Prediction using KNN")
st.write("Predict whether a person has heart disease or not")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("heart.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Split Features & Target
# -----------------------------
X = df.drop("target", axis=1)
y = df["target"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Train KNN Model
# -----------------------------
k = st.sidebar.slider("Select number of neighbors (K)", 1, 15, 5)

model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train_scaled, y_train)

# -----------------------------
# Model Accuracy
# -----------------------------
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

st.sidebar.write(f"### Model Accuracy: **{accuracy:.2f}**")

# -----------------------------
# User Input
# -----------------------------
st.subheader("Enter Patient Details")

age = st.number_input("Age", 20, 100, 45)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [0, 1])
restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope (0–2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thal (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)", [1, 2, 3])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                             thalach, exang, oldpeak, slope, ca, thal]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("✅ Heart Disease Detected")
    else:
        st.success("❌ No Heart Disease Detected")


