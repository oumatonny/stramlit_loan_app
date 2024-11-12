import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved models
log_model = joblib.load('logistic_regression_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')
svc_model = joblib.load('svc_model.pkl')

# Load the label encoders
label_encoders = {
    'Gender': joblib.load('label_encoder_gender.pkl'),
    'Married': joblib.load('label_encoder_married.pkl'),
    'Education': joblib.load('label_encoder_education.pkl'),
    'Self_Employed': joblib.load('label_encoder_self_employed.pkl'),
    'Area': joblib.load('label_encoder_area.pkl')
}

# Load model accuracies
model_accuracies = joblib.load('model_accuracies.pkl')

# Function to encode input data
def encode_input(data):
    for column in label_encoders:
        data[column] = label_encoders[column].transform([data[column]])[0]
    return data

# Streamlit app layout
st.title("Loan Eligibility Prediction")

# User input fields
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Marital Status", ["Yes", "No"])
dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
term = st.selectbox("Loan Term (months)", [120, 360])
credit_history = st.selectbox("Credit History", [0, 1])
area = st.selectbox("Area", ["Urban", "Semiurban", "Rural"])

# Select model for prediction
model_choice = st.selectbox("Select Model", ["Logistic Regression", "Random Forest", "Support Vector Classifier"])

# Button to predict
if st.button("Predict"):
    # Prepare the input data
    input_data = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Loan_Amount": loan_amount,
        "Term": term,
        "Credit_History": credit_history,
        "Area": area
    }
    
    input_df = pd.DataFrame(input_data, index=[0])
    
    # Encode the input data
    encoded_input = encode_input(input_df)

    # Make predictions based on selected model
    if model_choice == "Logistic Regression":
        model = log_model
    elif model_choice == "Random Forest":
        model = rf_model
    else:
        model = svc_model

    prediction = model.predict(encoded_input)

    # Display the result
    st.subheader("Prediction:")
    result = 'Approved' if prediction[0] == 1 else 'Not Approved'
    st.write(f"{model_choice} Prediction: {result}")

    # Feedback section
    st.subheader("Feedback on Prediction:")
    feedback = st.radio("Did you find this prediction helpful?", ("Thumbs Up 👍", "Thumbs Down 👎"))

    if feedback == "Thumbs Up 👍":
        st.success("Thank you for your feedback! We're glad you found the prediction helpful.")
    elif feedback == "Thumbs Down 👎":
        st.warning("Thank you for your feedback! We're sorry to hear that. Please let us know how we can improve.")

# Display model accuracies
st.subheader("Model Accuracies:")
accuracy_df = pd.DataFrame.from_dict(model_accuracies, orient='index', columns=['Accuracy'])
st.bar_chart(accuracy_df)

# Plotting Line Graph for Model Accuracies
st.subheader("Model Accuracy Line Graph:")
plt.figure(figsize=(10, 5))
sns.lineplot(data=accuracy_df.reset_index(), x='index', y='Accuracy', marker='o')
plt.title('Model Accuracies')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.grid()
st.pyplot(plt)

# Optionally, clear the figure after displaying
plt.clf()
