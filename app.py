import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("fake_job_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Fake Job Detection System")

job_description = st.text_area("Enter Job Details")

if st.button("Check Job"):
    data = vectorizer.transform([job_description])
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("⚠️ This job may be Fake")
    else:
        st.success("✅ This job looks Genuine")