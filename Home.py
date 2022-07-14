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


st.set_page_config(layout="wide")

# Remove whitespace from the top of the page and sidebar
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
st.image(file)

st.info("##### Machine Learning Model : by Godwin Nwalozie")

path = os.path.abspath(os.path.dirname(__file__))

# Load ML Models
@st.cache(suppress_st_warning=True, allow_output_mutation=True, persist= True)
def load_model ():
    model = joblib.load(os.path.join(path,"model_twitter.joblib"))
    return model



@st.cache(suppress_st_warning=True, allow_output_mutation=True, persist= True)
def load_data():
    data = pd.read_csv(os.path.join(path, 'twitter_cleaned.csv'), usecols=[0,1,2])
    return  data

    
   
st.write(""" #####  This ML model classifiies feedbacks or product reviews into negative, neutral and positive. \
    This can help the customer success or product teams to visualize and ascertain if a product \
    is doing well or in need of improvement. Are the customers happy in areas such as price, quality of service e.t.c\
  """)


st.markdown("")

plt.style.use('seaborn-ticks')
#st.write(data.sample(3))

with st.container():
    col1, col2 = st.columns(2)
    with col1:

        st.markdown("")
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
                    probab_neg = model.predict_proba([tweet])[:,0]
                    probab_neut = model.predict_proba([tweet])[:,1]
                    probab_pos = model.predict_proba([tweet])[:,2]
                    prediction = model.predict([tweet])[0] 
                    if prediction  == -1 :
                        prediction =  "Negative review 👎" 
                    elif  prediction == 0:
                        prediction = "Neutral Review   😐"
                    else:
                        prediction = "Positive Review  ⭐👍"
                    st.write(f" #### 【{prediction}】")
                    st.markdown(f"""##### Negative @ {probab_neg *100}% ⋆⋆  Neutral @{probab_neut*100}% ⋆⋆ Positive @ {probab_pos*100}% """)
                  
        

with col2:
    st.sidebar.title("Select plots")
    option = st.sidebar.radio('choose plot type', 
                      ("positive key words","negative key words",'count of tweets by airline', 
                       'distribution of sentiments(pie chart)', "sentiments by airline(bar graph)"))
   
    dir_name = os.path.abspath(os.path.dirname(__file__))
    file = Image.open(os.path.join(dir_name,"my photo.png"))
    st.sidebar.image(file,width= 170 )
    # Find me links
    kaggle=' 🔍Find me on Linkedin [link](https://www.linkedin.com/in/godwinnwalozie/)'
    st.sidebar.markdown(kaggle,unsafe_allow_html=True)
    git=' 🔍 Find me on Git [link](https://github.com/godwinnwalozie)'
    st.sidebar.markdown(git,unsafe_allow_html=True)
    kaggle=' 🔍Find me on Kaggle [link](https://www.kaggle.com/godwinnwalozie/code)'
    st.sidebar.markdown(kaggle,unsafe_allow_html=True)
    
    dataset =load_data()
    #wordcloud postive sentiments
    @st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
    def wordcloud_pos ():
        fig, ax = plt.subplots(dpi=1200) 
        super = dataset.loc[:,["tweets","airline_sentiment"]]
        text = "".join(super[super.airline_sentiment == "positive"].tweets)
        wc= WordCloud(max_words = 1000,background_color="whitesmoke", random_state=42,normalize_plurals=True).generate(text)
        plt.title("wordcloud - most recurring positive words", fontsize = 17)
        plt.axis("off")
        plt.tight_layout(pad=0)
        ax.imshow(wc,interpolation="bilinear")
        return fig
    plot1 = wordcloud_pos()
    
    #wordcloud negative sentiments
    @st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
    def wordcloud_neg ():
        fig, ax = plt.subplots(dpi=1200) 
        super = dataset.loc[:,["tweets","airline_sentiment"]]
        text = "".join(super[super.airline_sentiment == "negative"].tweets)
        wc= WordCloud(max_words = 500,background_color="whitesmoke",random_state= 42,normalize_plurals=True).generate(text)
        plt.title("wordcloud - most recurring negative words", fontsize = 17)
        plt.axis("off")
        plt.tight_layout(pad=0)
        ax.imshow(wc,interpolation="bilinear")
        return fig
    plot2 = wordcloud_neg()
    
    # count of customer tweets by airline'
    @st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
    def tweet_count ():
            fig, ax = plt.subplots(figsize =(10,4.5)) 
            dataset.loc[:,["airline","airline_sentiment"]].groupby("airline").count().plot(kind = "bar", ax= ax)
            plt.title("count of customer tweets by airline", fontsize = 13);
            return fig
    plot3 = tweet_count ()
    
    
    # sentiment % by airlines 
    @st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
    def perc_sentiment ():        
        fig, ax = plt.subplots(figsize =(2,4))
        dataset.airline_sentiment.value_counts().plot(kind = "pie",autopct = "%.0f%%", explode = (0.02,0.02,0.02) ,textprops={'fontsize': 6})
        plt.axis("off")       
        return fig
    plot4= perc_sentiment ()
    

    # sentiments by airline
    @st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
    def sent ():            
        fig, ax = plt.subplots(figsize =(10,4.5))        
        pd.crosstab(dataset.airline, dataset.airline_sentiment).plot( kind = "bar", ax = ax)
        plt.title("sentiment by airlines - bar graph" , fontsize = 13)
        plt.style.use('seaborn-darkgrid')
        return fig
    plot5= sent()




    if option  == "positive key words":
        plot1
    elif  option == "negative key words":
        plot2
    elif option == "count of tweets by airline":
        plot3
    elif option == "distribution of sentiments(pie chart)" :
        plot4
    else:
        plot5
        




