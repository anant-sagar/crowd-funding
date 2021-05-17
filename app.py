from pandas.core.frame import DataFrame
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle



st.title("Crowd funding")

st.image("images\crowdimage.jpg")


st.success('''The aim of this project is to construct such a model and also to analyse Kickstarter project 
data more generally, in order to help potential project creators assess whether or not Kickstarter is a good 
funding option for them, and what their chances of success are.''')




@st.cache()
def load_data(path):
    df= pd.read_csv(path)
    return df

df = load_data("dataset\ks-projects-201801.csv")

def load_model(path = 'crowdfunding_prediction.pk'):
    with open(path,"rb") as file :
        model = pickle.load(file)
    st.sidebar.info("Model Loaded Sucessfully.")
    return model

#First create a function that returns the success probability (using logistic regression)

def predict_project_success(data, model_dict):
    scaler = model_dict["scaler"]
    encoder = model_dict["encoder"]
    model = model_dict["model"]
    kmeans = model_dict["kmeans"]

    #Scale data

    X_numerical = data[['Days to deadline','usd_goal_real','backers']]
    scaled_X = pd.DataFrame(scaler.transform(X_numerical), columns = X_numerical.columns)

    scaled_X
    #Cluster data

    clusters = pd.DataFrame(kmeans.predict(scaled_X), columns = ['Cluster (k=4)'])

    # Concatenate categorical and numerical variables

    X = pd.concat([scaled_X, clusters, data[['main_category','country']].reset_index(drop = True)], axis = 1,)
    #Encode X

    encoded_X = encoder.transform(X)

    encoded_X

    #Generate prediction probability

    return (model.predict_proba(encoded_X)[0][1]) #Probability estimate that project is successful


if st.checkbox("About"):
    st.markdown("""We consider the problem of project success prediction on crowdfunding platforms. Despite the
information in a project profile can be of different modalities such as text, images, and metadata,
most existing prediction approaches leverage only
the text dominated modality. Nowadays rich visual
images have been utilized in more and more project
profiles for attracting backers, little work has been
conducted to evaluate their effects towards success
prediction. Moreover, meta information has been
exploited in many existing approaches for improving prediction accuracy. However, such meta information is usually limited to the dynamics after
projects are posted, e.g., funding dynamics such
as comments and updates. Such a requirement of
using after-posting information makes both project
creators and platforms not able to predict the outcome in a timely manner. In this work, we designed
and evaluated advanced neural network schemes
that combine information from different modalities
to study the influence of sophisticated interactions
among textual, visual, and metadata on project success prediction. To make pre-posting prediction
possible, our approach requires only information
collected from the pre-posting profile. Our extensive experimental results show that the image
features could improve success prediction performance significantly, particularly for project profiles
with little text information. Furthermore, we identified contributing elements.""")

if  st.checkbox("Make Prediction"):
    model_dict = load_model()

    name = st.text_input("Enter Name")

    col11, col12 = st.beta_columns(2)

    with col11:

        category= st.selectbox("Select a Category",['Narrative Film',
 'Music',
 'Restaurants',
 'Food',
 'Drinks',
 'Indie Rock',
 'Design',
 'Comic Books',
 'Art Books',
 'Fashion',
 'Childrenswear',
 'Theater',
 'Comics',
 'Webseries',
 'Animation',
 'Food Trucks',
 'Product Design',
 'Public Art',
 'Documentary',
 'Illustration',
 'Photography',
 'Tabletop Games',
 'Pop',
 'People',
 'Art',
 'Family',
 'Film & Video',
 'Accessories',
 'Rock',
 'Weaving',
 'Web',
 'Jazz',
 'Festivals',
 'Video Games',
 'Anthologies',
 'Publishing',
 'Shorts',
 'Gadgets',
 'Electronic Music',
 'Radio & Podcasts',
 'Apparel',
 'Metal',
 'Comedy',
 'Hip-Hop',
 'Painting',
 'Software',
 'Games',
 'World Music',
 'Photobooks',
 'Drama',
 'Hardware',
 'Young Adult',
 'Latin',
 'Mobile Games',
 'Flight',
 'Fine Art',
 'Action',
 'Playing Cards',
 'Makerspaces',
 'Fiction',
 "Children's Books",
 'Apps',
 'Audio',
 'Performance Art',
 'Ceramics',
 'Vegan',
 'Dance',
 'Poetry',
 'Graphic Novels',
 'Fabrication Tools',
 'Performances',
 'Sculpture',
 'Nonfiction',
 'Stationery',
 'Print',
 'Thrillers',
 'Events',
 'Classical Music',
 'Spaces',
 'Country & Folk',
 'Wearables',
 'Journalism',
 'Mixed Media',
 'Movie Theaters',
 'Technology',
 'Animals',
 'Digital Art',
 'Knitting',
 'Graphic Design',
 'DIY',
 'Community Gardens',
 'DIY Electronics',
 'Crafts',
 'Embroidery',
 'Camera Equipment',
 'Jewelry',
 'Fantasy',
 'Webcomics',
 'Horror',
 'Experimental',
 'Science Fiction',
 'Puzzles',
 'R&B',
 'Music Videos',
 'Architecture',
 'Video',
 'Plays',
 'Blues',
 'Faith',
 'Installations',
 'Small Batch',
 'Places',
 'Farms',
 'Footwear',
 'Zines',
 'Sound',
 '3D Printing',
 'Musical',
 'Workshops',
 'Woodworking',
 'Photo',
 'Immersive',
 'Letterpress',
 'Conceptual Art',
 'Live Games',
 'Ready-to-wear',
 'Academic',
 'Cookbooks',
 'Space Exploration',
 'Gaming Hardware',
 'Periodicals',
 "Farmer's Markets",
 'Nature',
 'Television',
 'Robots',
 'Typography',
 'Translations',
 'Calendars',
 'Textiles',
 'Pottery',
 'Interactive Design',
 'Video Art',
 'Candles',
 'Glass',
 'Pet Fashion',
 'Crochet',
 'Printing',
 'Punk',
 'Civic Design',
 'Kids',
 'Literary Journals',
 'Couture',
 'Bacon',
 'Romance',
 'Taxidermy',
 'Quilts',
 'Chiptune',
 'Residencies',
 'Literary Spaces'])

    with col12:
       main_category= st.selectbox("Select a Category",['Film & Video',
 'Music',
 'Food',
 'Design',
 'Comics',
 'Publishing',
 'Fashion',
 'Theater',
 'Art',
 'Photography',
 'Games',
 'Crafts',
 'Technology',
 'Journalism',
 'Dance'])


    col1, col2 = st.beta_columns(2)

    with col1:
       country = st.selectbox("Select your country",['US',
 'AU',
 'CA',
 'GB',
 'IT',
 'DE',
 'IE',
 'MX',
 'ES',
 'SE',
 'FR',
 'NZ',
 'CH',
 'AT',
 'NO',
 'BE',
 'DK',
 'HK',
 'NL',
 'SG'])
    with col2:
        currency= st.selectbox("Select your Currency",
['USD',
 'AUD',
 'CAD',
 'GBP',
 'EUR',
 'MXN',
 'SEK',
 'NZD',
 'CHF',
 'NOK',
 'DKK',
 'HKD',
 'SGD'])

    col3, col4 = st.beta_columns(2)
    
    with col3:
        Launched= st.date_input("Enter Launched Date")

    with col4:
        Deadline= st.date_input("Enter Deadline Date")

    col5, col6 = st.beta_columns(2)
    
    with col5:
        goal = st.number_input("Required Investment Goal",min_value=1000)

    with col6:
        pledged = st.number_input("Achived Goal",min_value=1000)

    col7, col8 = st.beta_columns(2)
    
    with col5:
        backers = st.number_input("Total number of peoples invested in the project",min_value=1)

        

    with col6:
        usd_pledged = st.number_input("Enter USD Pledge")

    col9, col10 = st.beta_columns(2)
    
    with col9:
        usd_pledged_real = st.number_input("Achived Goal Actual Ammount")
        
    with col10:  
        usd_goal_real =st.number_input("Enter USD Goal Real")

    col13, col14 = st.beta_columns(2)
    
    with col13:
        state_successful =st.number_input("State Successful")

    with col14:  
        days_deadline = st.number_input("Enter Remaining Days to Deadline")


    

    if st.sidebar.button('Predict')and name:
        
        features = np.array([[123,
                    name,
                    category,
                    main_category,
                    currency,
                    Deadline,
                    goal,
                    Launched,
                    pledged,
                    backers,
                    country,
                    usd_pledged,
                    usd_pledged_real,
                    usd_goal_real,
                    state_successful,
                    days_deadline]])

        cols = ['ID',
                'name',
                'category',
                'main_category',
                'currency',
                'deadline',
                'goal',
                'launched',
                'pledged',
                'backers',
                'country',
                'usd pledged',
                'usd_pledged_real',
                'usd_goal_real',
                'state_successful',
                'Days to deadline']

        df = pd.DataFrame(features,columns=cols)
        st.write(df.info())
        df = df.astype({'deadline':'datetime64',
                'launched':'datetime64'})

        result = predict_project_success(df, model_dict)
        st.sidebar.info(f"{result*100:.2f}% chances to success this project.")




if st.checkbox("Visualization"):
    visualization= st.selectbox("Training Data Graphs",['All Data Visualise',
    "Show the data of Success/failure rate",
    "Project category data",
    "Currency",
    "Distribution of time between launch and deadline",
    "USD goal real",
    "USD pledge real",])


    if  visualization=="All Data Visualise":
        st.image("images/visualise.png")

    if  visualization=="Project category data":
        st.image("images/categories.png")

    if  visualization=="Currency":
        st.image("images/currency.png")

    if  visualization=="Show the data of Success/failure rate":
        st.image("images/rate.png")

    if  visualization=="Hypertension":
        st.image("images/successbybakers.png")

    if  visualization=="Distribution of time between launch and deadline":
        st.image("images/timedistribution.png")
        
    if  visualization=="USD goal real":
        st.image("images/usdgoalreal.png")

    if  visualization=="USD pledge real":
        st.image("images/usdpledgereal.png")

    if  visualization=="Country Data":
        st.image("images/country.png")












        
        



