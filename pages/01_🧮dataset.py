import streamlit as st
import random
from Home import load_data  # importing data set from the app.py
import pandas as pd
import numpy as np


st.markdown("""
        <style>
               .css-18e3th9 {
                    padding-top: 0.5rem;
                    padding-bottom: 5rem;
                }
               .css-wjbhl0 {
                    padding-top: 3rem;
                    padding-bottom: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)

m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: cornflowerblue;
    color:#ffffff;
}
div.stButton > button:hover {
    background-color: #6F84FF;
    color:#ffffff
    }
</style>""", unsafe_allow_html=True)


dataset = load_data()
st.session_state['dataset'] = dataset

with st.container():
    row_count = len(dataset) 
    col_count = len(dataset.columns)
    st.title(" Detail of trained dataset and Model Estimator 📊")

    col1, col2, col3 = st.columns(3)
    col1.metric("Number of rows", row_count, "")
    col2.metric("Number of columns",col_count, "")
    col3.metric("Estimator", "KNeighborsClassifier", "")
    st.write("Twitter US Airline Sentiment")
    st.write("Data source  :https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment ")
    

st.header ("The trained dataset")


if st.button("click to randomize 5 rows"):
    random.random()
st.write(dataset.sample(5))

    
    





    



    
    
    

