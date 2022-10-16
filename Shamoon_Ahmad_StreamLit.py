import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
import plotly.express as px
import matplotlib
import nltk
nltk.download('all')
import seaborn as sns
from collections import Counter
from tokenize import tokenize, untokenize, NUMBER, STRING, NAME, OP
from io import BytesIO

st.title("Web App")

df = pd.read_csv("SMS_data_UTF.csv")

st.dataframe(df)

options = df['Label'].unique().tolist()

selected_options = st.sidebar.multiselect('What do you want?', options)

filtered_df = df[df["Label"].isin(selected_options)]

st.dataframe(filtered_df)

import nltk
import pandas as pd

top_N = 50

# replace all non-alphanumeric characters
df['sub_rep'] = df.Message_body.str.lower().str.replace('\W', ' ')

# tokenize
df['tok'] = df.sub_rep.apply(nltk.tokenize.word_tokenize)

words = df.tok.tolist()  # this is a list of lists
words = [word for list_ in words for word in list_]

# frequency distribution
word_dist = nltk.FreqDist(words)

# remove stopwords
stopwords = nltk.corpus.stopwords.words('english')
words_except_stop_dist = nltk.FreqDist(w for w in words if w not in stopwords)

# output the results
rslt = pd.DataFrame(word_dist.most_common(top_N), columns=['Word', 'Frequency'])

st.bar_chart(rslt, x='Word', y='Frequency', width=1000, height=1000, use_container_width=True)


df['Counts'] = df['Message_body'].map(df['Message_body'].value_counts())


st.line_chart(df, x='Date_Received', y='Counts', width =100, height =500, use_container_width=True)

