import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import re
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import json

def remove_emoji(tweet):
    """
    Desc : removes emojis from the tweets
    :param tweet: text input
    :return: string without emojis
    """
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', tweet)


def clean_tweets(tweet):
    """
    Desc: removes all the mentions, retweets, websites and tags from the text
    :return: cleaned text
    """
    if type(tweet) == np.float:
        return ''

    tweet = tweet.lower()
    tweet = re.sub("@[A-Za-z0-9_]+", "", tweet)
    tweet = re.sub("#", "", tweet)
    tweet = re.sub(r'https?:\/\/\S+', '', tweet)
    tweet = re.sub(r'RT[\s]+', '', tweet)
    tweet = remove_emoji(tweet)
    tweet = tweet.rstrip()
    tweet = tweet.lstrip()

    return tweet

def remove_stopwords(text):
    """
    Desc: removes stopwords from the text
    :param text: Input Tweet
    :return:  Input tweet with removed English stopwords
    """
    stopword = stopwords.words('english')
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    return text

@st.cache
def load_dataset(path):
    """

    :param path: path to the csv file
    :return: pandas dataframe
    """
    data = pd.read_csv(path, encoding='utf-8')
    return data



def tokenize_tweet(tokenizer,text,max_length):
    """

    :param tokenizer: tokenizer
    :param text: input text
    :param max_length: maximum length of the text after passing through tokenizer
    :return: tokenized and paaded sequence
    """
    text = tokenizer.texts_to_sequences(text)
    text_pad = pad_sequences(text, maxlen=max_length, padding='post')
    return text_pad

@st.cache
def import_model(model_name):
    """

    :param model_name: name of the model
    :return: loaded model
    """
    model_lstm = load_model(model_name)
    return model_lstm

@st.cache
def import_tokenizer(jsonfile):
    """

    :param jsonfile: filename of the tokenizer json file
    :return: tokenizer
    """
    with open(jsonfile) as f:
        data = json.load(f)
        tokenizerT = tf.keras.preprocessing.text.tokenizer_from_json(data)
    return tokenizerT


if __name__=='__main__':

    # loading the model
    model = load_model('model.h5')

    #loading the tokenizer
    tokenizer = import_tokenizer('tokenizer.json')

    # defining path to the data
    path = r'E:\Sentiment Analysis\sentiment\dataset.csv'


    # heading
    st.markdown("<h1 style='text-align: center; color: white;'>Sentiment Analysis and Detection</h1>",
                unsafe_allow_html=True)
    # subheading
    st.markdown(
        "<h5 style='text-align: center; color: white;'> Analyze the sentiment of a tweet </h5>",
        unsafe_allow_html=True)

    # Description of app
    st.write('With over 500 million tweets generated daily, Twitter is full of opinions, jokes, mentions, and more from people all over the world. All these tweets can be analyzed as sentiment data which analyze the feelings of the users sending those tweets.'
            'Creative use of advanced artificial intelligence techniques can be an effective tool for analyzing sentiments of tweets.')
    st.write('This is a webapp uses NLP and Deep Learning techniques to analyse sentiment of the tweets.'
            'The sentiment analysis of Twitter data has many organizational benefits such as understanding your brand more deeply, growing your influence, and improving customer service.'
            'Twitter sentiment analysis allows you to keep track of whats being said about your product or service on social media, and can help you detect angry customers or negative mentions before they escalate')

    # take input
    input_tweet = st.text_area("Enter Your Tweet Here : ")

    # Button on click
    if st.button('Predict'):
        try:
            # clean the tweet
            text = clean_tweets(input_tweet)

            # remove stopwords
            text = remove_stopwords(text)

            # tokenizing the input tweet
            max_length = 100
            text_pad = tokenize_tweet(tokenizer,text,max_length)

            # macking prediction on the new tweet
            prediction= tf.argmax(tf.argmax(model.predict(tf.expand_dims(text_pad,axis=-1))))
            prediction=prediction.numpy()
            if prediction==0:
                st.write('The sentiment of the tweet is Litigious')
            elif prediction==1:
                st.write('The sentiment of the tweet is Negative')
            elif prediction==2:
                st.write('The sentiment of the tweet is Positive')
            else:
                st.write('The sentiment of the tweet is Uncertain')

        except:
            st.write('Enter Valid Input')











