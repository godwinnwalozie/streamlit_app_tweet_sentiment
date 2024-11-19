# importing libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import joblib
import os
from wordcloud import WordCloud
import seaborn as sns
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import random
from sklearn.base import TransformerMixin
from sklearn.naive_bayes import MultinomialNB


st.set_page_config(layout="wide")

# Remove whitespace from the top of the page and sidebar
st.markdown("""
        <style>
                .css-18e3th9 {
                    padding-top: 0.9rem;
                    padding-bottom: 0.2rem;
                }
                .css-hxt7ib {
                    padding-top: 1rem;
                    padding-bottom: 1rem;
                }
   
        </style>
        """, unsafe_allow_html=True)


# button styling
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




dir_name = os.path.abspath(os.path.dirname(__file__))
file = Image.open(os.path.join(dir_name,"title_image-2d.png"))
st.image(file,width=600)

st.write(" ###### Machine Learning Model : by Godwin Nwalozie")

path = os.path.abspath(os.path.dirname(__file__))

# Load ML Models
## @st.cache(suppress_st_warning=True, allow_output_mutation=True, persist= True)
def load_model ():
    model = joblib.load(os.path.join(path,"model_twitter.joblib"))
    return model


## @st.cache(suppress_st_warning=True, allow_output_mutation=True, persist= True)
def load_data():
    data = pd.read_csv(os.path.join(path, 'twitter_cleaned.csv'), usecols=[0,1,2])
    return  data


dataset =load_data()
## initialize session state
st.session_state['dataset'] = dataset


st.markdown("""
<style>
.big-font {
    font-size:17px !important;
    color :black;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">This Machine Learning Sentiment Analysis Visualization is a \
            graphical representation of customer sentiment derived from analyzing textual \
            data using machine learning models. It identifies and categorizes feedback into sentiments such as positive, neutral, or negative by processing large volumes of text \
            data from sources like social media, surveys, or reviews.', 
    unsafe_allow_html=True)  


#st.markdown("")

plt.style.use("seaborn-v0_8")
#st.write(data.sample(3))

with st.container():
    col1, col2 = st.columns(2)
    with col1:

        st.markdown("***")
        with st.container():
            
            
            st.write("##### Enter a sample tweet or product review in the text box below")
            
            tweet = st.text_input('delete review to input yours', 'sample: the customer service is very poor and they delayed in fixing my issues ') 
            

            if st.button('click to make a prediction 👈'):
                if tweet == "" :
                    counter = len(tweet)  
                    st.markdown(f" character counter: {counter}")
                    st.error(" ##### ...empty ! 😀 input some text")
                                      
                elif len(tweet) < 25:
                    st.error(" #####  😔 enter more characters")
                    counter = len(tweet)  
                    st.markdown(f" character counter: {counter}")
                    
                
                else:
                    model = load_model()
                    probab = model.predict_proba([tweet])
                    probab_neg = np.round(model.predict_proba([tweet]),3)[:,0]
                    probab_neut = np.round(model.predict_proba([tweet]),3)[:,1]
                    probab_pos = np.round(model.predict_proba([tweet]),3)[:,2]
                    prediction = model.predict([tweet])[0] 
                    if prediction  == -1 :
                        prediction =  "Negative review 👎" 
                    elif  prediction == 0:
                        prediction = "Neutral Review   😐"
                    else:
                        prediction = "Positive Review  ⭐👍"
                        st.balloons()
                    st.write(f" #### 【{prediction}】")
                    st.markdown(f"""##### Negative @ {probab_neg *100}% ⋆⋆  Neutral @{probab_neut*100}% ⋆⋆ Positive @ {probab_pos*100}% """)
                  
        

with col2:
    st.sidebar.title("Select plots")
    option = st.sidebar.radio('choose plot type', 
                      ("dataset","positive key words","negative key words",'count of tweets by airline', 
                       'distribution of sentiments(pie chart)', "sentiments by airline(bar graph)"))  
    
    def show_dataset ():
        if st.button("randomize dataset"):
            random.random()
        st.write(dataset.sample(7))
   
        
    #wordcloud postive sentiments
  
    @st.cache_data(hash_funcs={matplotlib.figure.Figure: lambda _: None})
    def wordcloud_pos ():
        fig, ax = plt.subplots() 
        super = dataset.loc[:,["tweets","airline_sentiment"]]
        text = "".join(super[super.airline_sentiment == "positive"].tweets.astype(str))
        wc= WordCloud(max_words = 1000,background_color="black", max_font_size=100, scale=10,\
    relative_scaling=.6,random_state=42,normalize_plurals=True).generate(text)
        plt.title("wordcloud - most recurring positive words", fontsize = 17)
        plt.axis("off")
        #plt.tight_layout(pad=0)
        ax.imshow(wc,interpolation="bilinear")
        return fig
    plot1 = wordcloud_pos()



    
    #wordcloud negative sentiments
    @st.cache_data(hash_funcs={matplotlib.figure.Figure: lambda _: None})
    def wordcloud_neg ():
        fig, ax = plt.subplots() 
        super = dataset.loc[:,["tweets","airline_sentiment"]]
        text = "".join(super[super.airline_sentiment == "negative"].tweets.astype('str'))
        wc= WordCloud(max_words = 500,background_color="black", max_font_size=100, scale=10,\
    relative_scaling=.6,random_state=42,normalize_plurals=True).generate(text)
        plt.title("wordcloud - most recurring negative words", fontsize = 17)
        plt.axis("off")
        #plt.tight_layout(pad=0)
        ax.imshow(wc,interpolation="bilinear")
        return fig
    plot2 = wordcloud_neg()


    
    # count of customer tweets by airline'
    @st.cache_resource(hash_funcs={matplotlib.figure.Figure: lambda _: None})
    def tweet_count ():
        fig = px.bar(dataset.loc[:,["airline","airline_sentiment"]].groupby("airline").count(),\
            title="Count of tweets by airline")
        return fig
    plot3 = tweet_count()
    
    
    # sentiment % by airlines 
    def pie_perc ():
        fig = px.pie(dataset.airline_sentiment.value_counts(), values = dataset.airline_sentiment.value_counts().values,
                    names= dataset.airline_sentiment.value_counts().index,title="% distribution of sentiments(pie chart)",hole=0.5 )
        fig.update_traces(textposition='outside', textinfo='percent+label')
        return fig
    plot4 = pie_perc()
    
    
    # sentiments by airline
    @st.cache_resource(hash_funcs={matplotlib.figure.Figure: lambda _: None})
    def sent ():            
        fig = px.bar(pd.crosstab(dataset.airline, dataset.airline_sentiment),title="Sentiments by airline(bar graph)",barmode='group')
        return fig
    plot5= sent()



    if option == "dataset":
        show_dataset()
    elif option  == "positive key words":
        plot1
    elif  option == "negative key words":
        plot2
    elif option == "count of tweets by airline":
        plot3
    elif option == "distribution of sentiments(pie chart)" :
        plot4
    else:
        plot5

  
dir_name = os.path.abspath(os.path.dirname(__file__))
file = Image.open(os.path.join(dir_name,"mazi.png"))
st.sidebar.image(file, )
# Find me links
kaggle=' 🔍Find me on Linkedin [link](https://www.linkedin.com/in/godwinnwalozie/)'
st.sidebar.markdown(kaggle,unsafe_allow_html=True)
git=' 🔍 Find me on Git [link](https://github.com/godwinnwalozie)'
st.sidebar.markdown(git,unsafe_allow_html=True)
kaggle=' 🔍Find me on Kaggle [link](https://www.kaggle.com/godwinnwalozie/code)'
st.sidebar.markdown(kaggle,unsafe_allow_html=True)
    




