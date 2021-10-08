#TODO : add to lower methods
# method to clean text 
import streamlit as st
import pickle
from sklearn.metrics import accuracy_score

st.sidebar.text("this is side bar")
model = pickle.load(open("model","rb"))
# select_model = 
countvectorizer = pickle.load(open("countvectorizer","rb"))
st.header('Stock Sentiment Analysis')
text_input =  st.text_area(label='Enter Text',height=200)
text = countvectorizer.transform([text_input])
if len(text_input)> 0:
    st.text('Sentiment is : ')
    sent = "positive" if model.predict(text)[0] else "stock might not increase" 
    st.bar_chart(model.predict_proba(text)[0])
    st.write("{}".format(sent))
    # st.write(" algo : Random Forest {}".format(accuracy_score()))
else:
    st.write("input is Empty or please Enter input . ")
