import streamlit as st
import joblib

# Load the vectorizer
vectorizer = joblib.load("vector.pkl")

# Load the model
model = joblib.load("model.pkl")

# Create a title
st.title("Fake News Classification app")

# Input text
text = st.text_area("Enter the news article:")

# Button to run inference
if st.button("Predict"):

    # Convert the text to a vector
    vector = vectorizer.transform([text])

    # Predict the label
    label = model.predict(vector)[0]

    # Display the label
    if label == 0:
        st.write("The news article is fake.")
    else:
        st.write("The news article is real.")
