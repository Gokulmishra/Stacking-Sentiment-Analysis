import streamlit as st
import pickle
st.sidebar.text("this is side bar")
#model = pickle.load(open("model","rb"))
countvectorizer = pickle.load(open("countvectorizer","rb"))
st.header('Stock Sentiment Analysis')
st.write("Description : ")
text_input =  st.text_area(label='Enter Text',height=200)
text = countvectorizer.transform([text_input])
classifier=st.radio("select Classifier",('Random Forest', 'K Neighbors','Logistic Regression','SVM'))
def loadModel(ModelName):
    if(ModelName=="Random Forest"):
        return pickle.load(open("model","rb"))
    if(ModelName=="K Neighbors"):
        return pickle.load(open("K_neighbors","rb"))
    if(ModelName=="Logistic Regression"):
        return pickle.load(open("logisticR","rb"))
    if(ModelName=="SVM"):
        return pickle.load(open("modelsvm","rb"))
model=loadModel(classifier)
if len(text_input)> 0:
    st.text('Sentiment is : ')
    sent = "Stock will rise" if model.predict(text)[0] else "Stock will fall" 
   # if(ModelName != "SVM"): 
    try:
        st.bar_chart(model.predict_proba(text)[0]) 
    except:
        print(" ")
    st.markdown("<h3 style='text-align: center; color: #FFFFFF;'>{}</h3>".format(sent), unsafe_allow_html=True)
    
