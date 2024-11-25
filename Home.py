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

text_input_area = '''
    <style>
        div.css-1om1ktf.e1y61itm0 {
          width: 800px;
        }
        textarea.st-cl {
          height: 400px;
        }
    </style>
    '''


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
file = Image.open(os.path.join(dir_name,"image header- sentiment.png"))
st.image(file,width=400)

st.write(" 𝗗𝗲𝘃𝗲𝗹𝗼𝗽𝗲𝗱 𝗕𝘆: 𝗚𝗼𝗱𝘄𝗶𝗻 𝗡𝘄𝗮𝗹𝗼𝘇𝗶𝗲")


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
    font-size:14px !important;
    color :black;
}
</style>
""", unsafe_allow_html=True)

st.write('<p class="big-font">A machine learning model for airline sentiment analysis processes customer feedback, such as social media posts, reviews, and surveys, to understand public sentiment toward an airline. The model uses Natural Language Processing (NLP) techniques to classify text into categories like positive, neutral, or negative sentiments.', 
    unsafe_allow_html=True)  

st.write("𝗗𝗮𝘁𝗮𝘀𝗲𝘁 𝗧𝗿𝗮𝗶𝗻𝗲𝗱 : 𝗔𝗯𝗼𝘂𝘁 20,000 𝘁𝘄𝗲𝗲𝘁𝘀 𝗳𝗿𝗼𝗺 𝗔𝗶𝗿𝗹𝗶𝗻𝗲 𝗰𝘂𝘀𝘁𝗼𝗺𝗲𝗿𝘀")
st.markdown("****")


plt.style.use("seaborn-v0_8")
## st.write(data.sample(3))

with st.container():
    col1, col2 = st.columns([0.39,0.61])
    with col1:

        #st.markdown("***")
        with st.container():
            
            
            st.write('###### Enter a sample tweet or product review to test',unsafe_allow_html=True)
            tweet = st.text_input('', 'sample: the customer service is very poor and they delayed in fixing my issues ') 
            

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
    st.sidebar.title("Select Plots")
    option = st.sidebar.radio('choose plot type', 
                      ("Dataset","+ve wordcloud","-ve wordcloud",'# of tweets by airline', 
                       'Sentiments(pie chart)', "Sentiments by airline(bar graph)",))  
    



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
        wc= WordCloud(max_words = 2000,background_color="black", max_font_size=80, scale=10,\
    relative_scaling=.6,random_state=42,normalize_plurals=True).generate(text)
        plt.title("Wordcloud | Most recurring positive words", fontsize = 17)
        plt.axis("off")
        #plt.tight_layout(pad=0)
        ax.imshow(wc,interpolation="bilinear")
        plt.savefig("result.png",dpi=300)
        return fig
    plot1 = wordcloud_pos()

    
    #wordcloud negative sentiments
    @st.cache_data(hash_funcs={matplotlib.figure.Figure: lambda _: None})
    def wordcloud_neg ():
        fig, ax = plt.subplots() 
        super = dataset.loc[:,["tweets","airline_sentiment"]]
        text = "".join(super[super.airline_sentiment == "negative"].tweets.astype('str'))
        wc= WordCloud(max_words = 2000,background_color="black", max_font_size=80, scale=10,\
    relative_scaling=.6,random_state=42,normalize_plurals=True).generate(text)
        plt.title("Wordcloud | Most recurring negative words", fontsize = 17)
        plt.axis("off")
        #plt.tight_layout(pad=0)
        ax.imshow(wc,interpolation="bilinear")
        plt.savefig("result.png",dpi=300)
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
    elif option  == "+ve wordcloud":
        plot1
    elif  option == "-ve wordcloud":
        plot2
    elif option == "# of tweets by airline":
        plot3
    elif option == "Sentiments(pie chart)" :
        plot4
    else:
        plot5

  
dir_name = os.path.abspath(os.path.dirname(__file__))
file = Image.open(os.path.join(dir_name,"mazi.png"))
st.sidebar.image(file,width=150 )
# Find me links
kaggle=' 🔍Find me on Linkedin [link](https://www.linkedin.com/in/godwinnwalozie/)'
st.sidebar.markdown(kaggle,unsafe_allow_html=True)
git=' 🔍 Find me on Git [link](https://github.com/godwinnwalozie)'
st.sidebar.markdown(git,unsafe_allow_html=True)
kaggle=' 🔍Find me on Kaggle [link](https://www.kaggle.com/godwinnwalozie/code)'
st.sidebar.markdown(kaggle,unsafe_allow_html=True)
    




