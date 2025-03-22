import streamlit as st
import pickle
import pickle
import os

# Use a relative path to load the model and vectorizer
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model.pkl')
vectorizer_path = os.path.join(current_dir, 'vectorizer.pkl')

model = pickle.load(open(model_path, 'rb'))
vectorizer = pickle.load(open(vectorizer_path, 'rb'))


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
