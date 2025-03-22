import streamlit as st
import pickle

# Load the trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# App title
st.title("SMS Spam Detection")
st.write("Enter an SMS message below to check if it's Spam or Not Spam.")

# Input box for the message
input_message = st.text_input("Enter your message:")

if st.button("Predict"):
    if input_message.strip():
        # Preprocess and predict
        input_vectorized = vectorizer.transform([input_message])
        prediction = model.predict(input_vectorized)[0]
        result = "Spam" if prediction == 1 else "Not Spam"
        st.subheader(f"Prediction: {result}")
    else:
        st.warning("Please enter a valid SMS message.")
