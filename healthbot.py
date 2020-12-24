import streamlit as st

st.write("""
# Health Bot
This is a healthbot curated by college student for eductional purposes only.
this will communicate with the user and ask about their diseases and symptoms faced by them
""")


name = st.text_input('Please tell me your Name.')
age= st.number_input("Please enter your age.")


st.write(f"Hello,{name} myself healthbot RoRo and i will be helping you out today. ")


import pandas as pd
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

import csv
import seaborn as sns

def get_data():
    df = pd.read_csv('mySddf.csv')
    return df

lp=pd.read_csv('mySddf.csv')

lp.isnull().sum()
list_drop=['disease']
lp.drop(list_drop,axis=1, inplace= True)
#lp.head()

def data_prepare():# all this work for creating dictionary
    col = ['Unnamed: 0','Disease','Symptom_1','Symptom_2','Symptom_3', 'Description', 'Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']
    y= get_data() #provides access to file resources distributed with the package.
    y = y[col]
    y = y[pd.notnull(y['Description'])] # removes the row if found null
    y.columns = ['Unnamed: 0','Disease','Symptom_1','Symptom_2','Symptom_3', 'Description', 'Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']
    y['category_id'] = y['Disease'].factorize()[0]
    #print(y['category_id'])
    category_id_df = y[['Disease', 'category_id']].drop_duplicates().sort_values('category_id')#removes duplicates
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['Disease','category_id']].values)#This will add the column in our dataframe
    return y

def naive_algo(choice):
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    df=data_prepare()
    features = tfidf.fit_transform(df.Disease).toarray()
    labels = df.category_id
    features.shape
    X_train, X_test, y_train, y_test = train_test_split(df['Disease'], df[choice], random_state = 0)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = MultinomialNB().fit(X_train_tfidf, y_train)
    return clf,count_vect

def predict(Disease,choice):
    clf,count_vect=naive_algo(choice)
    intent=clf.predict(count_vect.transform([Disease]))
    intent=str(intent).strip("['']")
    print(intent)
    return intent


ques= st.text_input(f"Now, {name} tell me about the name of the illness you are facing")

for i in range(8):
    if(i==0):
        st.write("Symptoms are: ")
        x=predict(ques,'Symptom_1')
        st.write(intent=str(x).strip("['']"))
    elif(i==1):
        x=predict(ques,'Symptom_2')
        intent=str(x).strip("['']")
    elif(i==2):
        x=predict(ques,'Symptom_3')
        intent=str(x).strip("['']")
        print('\n')
    elif(i==3):
        print("Description:")
        x=predict(ques,'Description')
        intent=str(x).strip("['']")
        print('\n')
    elif(i==4):
        print("Precautions")
        x=predict(ques,'Precaution_1')
        intent=str(x).strip("['']")
    elif(i==5):
        x=predict(ques,'Precaution_2')
        intent=str(x).strip("['']")
