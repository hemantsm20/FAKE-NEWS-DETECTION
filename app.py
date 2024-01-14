# import streamlit as st
# import pickle
# import re
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from sklearn.feature_extraction.text import TfidfVectorizer
# port_stem = PorterStemmer()
# vectorization = TfidfVectorizer()

# vector_form = pickle.load(open('vector.pkl', 'rb'))
# load_model = pickle.load(open('model.pkl', 'rb'))

# def stemming(content):
#     con=re.sub('[^a-zA-Z]', ' ', content)
#     con=con.lower()
#     con=con.split()
#     con=[port_stem.stem(word) for word in con if not word in stopwords.words('english')]
#     con=' '.join(con)
#     return con

# def fake_news(news):
#     news=stemming(news)
#     input_data=[news]
#     vector_form1=vector_form.transform(input_data)
#     prediction = load_model.predict(vector_form1)
#     return prediction



# if __name__ == '__main__':
#     st.title('Fake News Classification app ')
#     st.subheader("Input the News content below")
#     sentence = st.text_area("Enter your news content here", "",height=200)
#     predict_btt = st.button("predict")
#     if predict_btt:
#         prediction_class=fake_news(sentence)
#         print(prediction_class)
#         if prediction_class == [0]:
#             st.success('Reliable')
#         if prediction_class == [1]:
#             st.warning('Unreliable')




import streamlit as st
import joblib

# Load the vectorizer
vectorizer = joblib.load("vector.pkl")

# Load the model
model = joblib.load("model.pkl")

# Create a title
st.title("Predict Fake News")

# Input text
text = st.text_input("Enter the news article:")

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
