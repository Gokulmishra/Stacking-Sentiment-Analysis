#TODO : add to lower methods
# method to clean text 
import streamlit as st
import pickle
st.sidebar.text("this is side bar")
#model = pickle.load(open("model","rb"))
countvectorizer = pickle.load(open("countvectorizer","rb"))
st.header('Stock Sentiment Analysis')
st.write("Description : This project is about taking non-quantifiable data such as news articles about a company’s recent financial activity and predicting its future stock trend with news sentiment classification ")
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
    
    try:
        st.bar_chart(model.predict_proba(text)[0])
        
    except:
        print(" ")
    st.text('1:{} '.format(model.predict_proba(text)[0][1]))  
    st.text('0:{} '.format(model.predict_proba(text)[0][0]))  
    if model.predict_proba(text)[0][1]>0.5:
        st.markdown("<h3 style='text-align: center; color: #FFFFFF;'>{}</h3>".format("Stock will rise"), unsafe_allow_html=True)
    else: 
        st.markdown("<h3 style='text-align: center; color: #FFFFFF;'>{}</h3>".format("Stock will fall"), unsafe_allow_html=True)
   
    
