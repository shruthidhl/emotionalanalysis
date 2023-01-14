from flask import Flask

import streamlit as st
from PIL import Image
import pickle
import numpy
import pandas as pd
import os
import sys
from streamingdata import fetchStreamingData
from cleanscopedata import clean_scope
# app = Flask(__name__)

# @app.route('/prediction')


def predict():
    projectdir=os.getcwd()
    # features = [ i for i in range(0,500)]
    # final_features= [numpy.array(features)]
    #fetch Streaming data 
    # search_words = '#climatechange'
    # date_since = "2018-11-16"
    #path_tsa=projectdir+"/tsa.png"
    #image = Image.open(path_tsa)
    #st.image(image)
    st.title('Welcome to Twitter Sentiment Prediction!')
    search_words = st.text_input("Search a hashtag!", "Please enter a keyword to search")
    date_since = st.date_input("Select a date to view the tweets from", value=None, min_value=None, max_value=None)

   
    fetchdata=fetchStreamingData(search_words,date_since)
    
    #clean and featurize
     
    path=projectdir+"/data/raw/"
    scope_filepath = os.path.join(path, "raw_scope.csv")
    tweet_text_df=pd.read_csv(scope_filepath,index_col=False)
    prepare=clean_scope(tweet_text_df)
    
    path=projectdir+"/data/features/"
    # X_predict_features=pd.read_csv(path+"scope_features.csv",index_col=False)
    # print('X_predict_features',X_predict_features.head())
    # print('X_predict_features',X_predict_features.shape)
    feature_filepath = os.path.join(path, "scope_features.pkl")
    X_predict_features = pickle.load(open(feature_filepath,"rb"))

    #load models    
    path=projectdir+"/models/"
    polarity_filepath = os.path.join(path, "polarityclassifier.pkl")
    sentiment_filepath = os.path.join(path, "sentimentclassifier.pkl")
    polarity_class_model = pickle.load(open(polarity_filepath,"rb"))
    sentiment_class_model= pickle.load(open(sentiment_filepath,"rb"))


    #predict
    print('prediction')
    print('Hello world!', file=sys.stderr)
    row,columns=X_predict_features.shape
    polarity = pd.DataFrame(polarity_class_model.predict( X_predict_features))
    sentiment = pd.DataFrame(sentiment_class_model.predict( X_predict_features))
    df=pd.DataFrame()
    df=pd.concat([tweet_text_df,polarity,sentiment],axis=1)
    print(df.head())
    
    st.write(df.head(20))
    print('polarity')
    return 'Hi'


# @app.route('/')
# def home_page():
#     return "welcome to prediction home"


if __name__ == "__main__":
 # app.run(debug=True,host='0.0.0.0',port='8080')
  predict()
