import streamlit as st
import pickle
tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))
import re  #regular expression
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def simple_stemmer(word):           #stemming means treat words like dance ,dancing,danced as same as dance only
    suffixes = ['ing', 'ly', 'ed', 'ious', 'ies', 'ive', 'es', 's', 'ment']
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            return word[:-len(suffix)]
    return word

def transform_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = re.sub(r'[^a-z0-9\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    words = [simple_stemmer(word) for word in words]
    return " ".join(words)  # <-- return as string

st.title('Email/SMS Spam Classifier')
input_sms=st.text_area("Enter the message")
if st.button('Predict'):
    transform_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transform_sms])
    result = model.predict(vector_input)[0]
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")



