import pickle

import numpy as np
import streamlit as st

# Modelni yuklash
with open("diabetes.pkl", "rb") as file:
    data = pickle.load(file)

# Interfeys sarlavhasi
st.title("Diabetes Prediction App")
st.write("Logistik Regressiya modeli asosida diabetes ehtimolini bashorat qilish.")

# Modelni o‘rgatish
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X = data.drop("Outcome", axis=1)
y = data["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Foydalanuvchi kiritish maydonlari
preg = st.number_input("Homiladorliklar soni", min_value=0, max_value=20, step=1)
glucose = st.number_input("Glukozaning darajasi", min_value=0, max_value=200, step=1)
bp = st.number_input("Qon bosimi", min_value=0, max_value=150, step=1)
skin = st.number_input("Teri qalinligi", min_value=0, max_value=100, step=1)
insulin = st.number_input("Insulin darajasi", min_value=0, max_value=900, step=1)
bmi = st.number_input("BMI (tana massasi indeksi)", min_value=0.0, max_value=70.0, step=0.1)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01)
age = st.number_input("Yosh", min_value=0, max_value=120, step=1)

# Bashorat qilish
if st.button("Natija olish"):
    features = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.write("Diabetes ehtimoli BOR.")
    else:
        st.write("Diabetes ehtimoli YO'Q.")
