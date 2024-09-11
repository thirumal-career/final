import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit as st
from datetime import datetime
#import mysql.connector

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Preprocess text (you can adjust this function as needed)
def preprocess_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    return " ".join(tokens)

# Analyze sentiment using VADER
def analyze_sentiment_vader(text):
    cleaned_text = preprocess_text(text)
    sentiment_scores = analyzer.polarity_scores(cleaned_text)
    return sentiment_scores

# Streamlit app
def streamlit_app():
    tab1, tab2 = st.tabs(["Home", "Sentiment Analysis"])
    with tab1:
        st.markdown('<h1 style="text-align: center; color: red;">GUVI SENTIMENT ANALYSIS</h1>', unsafe_allow_html=True)
        name = st.text_input("Please enter your name", "")
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        if st.button("Login"):
            mycursor.execute(
                "INSERT INTO cust_info (name, login) VALUES (%s, %s)",
                (name, formatted_datetime)
            )
            mydb.commit()
            st.success('Data migrated to RDS-Mysql server!', icon="✅")

    with tab2:
        sentence = st.text_input("Please enter the sentence", "")
        if st.button("Sentiment Analysis"):
            sentiment_scores = analyze_sentiment_vader(sentence)
            st.write(f"Sentiment scores: {sentiment_scores}")

            scores = []
            scores.append(sentiment_scores['neg'])
            scores.append(sentiment_scores['neu'])
            scores.append(sentiment_scores['pos'])
            max_scores = max(scores)
            max_scores_index = scores.index(max_scores)
            if max_scores_index == 0:
                st.subheader(":red[Sentiment of the sentence is] Negative")
            elif max_scores_index == 1:
                st.subheader(":red[Sentiment of the sentence is] Neutral")
            else:
                st.subheader(":red[Sentiment of the sentence is] Positive")


#mydb = mysql.connector.connect(
   # host="please provide rds host id",
  #  user="admin",
 #   password="please provide rds password ",
#    port="please provide port number",   )
#mycursor = mydb.cursor(buffered=True)

#mycursor.execute("create database if not exists sentimentanalysis")
#mycursor.execute("use sentimentanalysis")
#mycursor.execute("create table if not exists cust_info (name varchar(255) ,login DATETIME)")


streamlit_app()
