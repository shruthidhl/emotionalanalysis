from flask import Flask
import pickle
import numpy
import pandas as pd
import os
import sys
#from streamingdata import fetchStreamingData
from cleanscopedata import clean_scope
app = Flask(__name__)

@app.route('/prediction')
def predict():
    # features = [ i for i in range(0,500)]
    # final_features= [numpy.array(features)]
    #fetch Streaming data 
    search_words = '#climatechange'
    date_since = "2018-11-16"
    projectdir=os.getcwd()
   # fetchdata=fetchStreamingData(search_words,date_since)
    
    #clean and featurize
     
    path=projectdir+"/data/raw/"
    scope_filepath = os.path.join(path, "raw_scope.csv")
    tweet_text_df_origin=pd.read_csv(scope_filepath,index_col=False)
    tweet_text_df=tweet_text_df_origin.copy()
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
    polarity.rename({0:"polarity"},inplace=True,axis=1)
    sentiment.rename({0:"sentiment"},inplace=True,axis=1)
    print(polarity.head())
    print(sentiment.head())
    print(tweet_text_df_origin.head())
    tweets=pd.DataFrame(tweet_text_df_origin['content'])
    df=pd.concat([tweets,polarity,sentiment],axis=1)
    print(df.head())
    return df

@app.route('/')
def home_page():
    return "welcome to prediction home"

if __name__ == "__main__":
  app.run(debug=True,host='0.0.0.0',port='8080')
  predict()
    


