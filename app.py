import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title("🎓 Student Performance Predictor")

st.sidebar.header("Enter Student Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
study_hours = st.sidebar.slider("Study Hours per Day", 1, 10, 3)
attendance = st.sidebar.slider("Attendance (%)", 50, 100, 75)
internet = st.sidebar.selectbox("Internet Access", ["Yes", "No"])
past_score = st.sidebar.slider("Previous Exam Score", 40, 100, 60)
extra_activities = st.sidebar.selectbox("Extra Activities", ["Yes", "No"])

gender = 1 if gender == "Male" else 0
internet = 1 if internet == "Yes" else 0
extra_activities = 1 if extra_activities == "Yes" else 0

input_data = pd.DataFrame([[
    gender, study_hours, attendance, internet, past_score, extra_activities
]], columns=['Gender', 'Study_Hours', 'Attendance', 'Internet', 'Past_Score', 'Extra_Activities'])

model = RandomForestClassifier()
X_dummy = pd.DataFrame(np.random.randint(0, 2, size=(100, 6)), columns=input_data.columns)
y_dummy = np.random.randint(0, 2, size=(100,))
model.fit(X_dummy, y_dummy)

if st.button("Predict Result"):
    prediction = model.predict(input_data)[0]
    result = "✅ Pass" if prediction == 1 else "❌ Fail"
    st.subheader(f"Prediction: {result}")
