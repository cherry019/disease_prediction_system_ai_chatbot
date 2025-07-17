import streamlit as st
import pandas as pd
import joblib

# ------------------ Load Model and Data ------------------ #
model = joblib.load("disease_model.pkl")
le = joblib.load("label_encoder.pkl")
unique_symptoms = joblib.load("unique_symptoms.pkl")

desc_df = pd.read_csv("symptom_Description.csv")
prec_df = pd.read_csv("symptom_precaution.csv")

# ------------------ App Title ------------------ #
st.set_page_config(page_title="Disease Predictor", layout="centered")
st.title("🩺 AI-Based Disease Prediction Chatbot")
st.markdown("Select symptoms below and click **Predict** to get your diagnosis.")

# ------------------ Symptom Selection ------------------ #
selected_symptoms = st.multiselect("🧾 Select your symptoms:", unique_symptoms)

# ------------------ Prediction ------------------ #
if st.button("🔮 Predict Disease"):
    if selected_symptoms:
        input_vector = [1 if symptom in selected_symptoms else 0 for symptom in unique_symptoms]
        prediction = model.predict([input_vector])
        disease = le.inverse_transform(prediction)[0]

        # Fetch description
        desc_row = desc_df[desc_df["Disease"].str.lower() == disease.lower()]
        description = desc_row["Description"].values[0] if not desc_row.empty else "No description available."

        # Fetch precautions
        pre_row = prec_df[prec_df["Disease"].str.lower() == disease.lower()]
        precautions = [pre_row[f"Precaution_{i}"].values[0] for i in range(1, 5)] if not pre_row.empty else ["No precautions found."]

        # Display results
        st.success(f"🎯 Predicted Disease: **{disease}**")
        st.markdown(f"**📖 Description:** {description}")
        st.markdown("**🛡️ Precautions:**")
        for p in precautions:
            st.markdown(f"- {p}")
    else:
        st.warning("⚠️ Please select at least one symptom before predicting.")
