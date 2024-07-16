import streamlit as st
import pickle
import time

st.title("Twitter Sentiment Analysis")

# Load model
with open('saved_LRmodel.sav', 'rb') as f:
    model = pickle.load(f)

# Load the TfidfVectorizer
with open('saved_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

tweet = st.text_input("Enter your text")

submit = st.button('Predict')

if submit:
    # Transform the input using the loaded TfidfVectorizer
    transformed_tweet = vectorizer.transform([tweet])

    start = time.time()
    prediction = model.predict(transformed_tweet)
    end = time.time()

    st.write('Prediction time taken: ', round(end - start, 2), 'seconds')
    st.write("Predicted sentiment is: ", 'Negative' if prediction[0] == 0 else 'Positive')
    # to run this code
    # 1. open terminal
    # 2. run the following command "streamlit run TSA_app.py"
    # 3. type or paste any tweet in the input box and get the prediction
