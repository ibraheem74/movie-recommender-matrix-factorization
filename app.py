import streamlit as st
import joblib
import pandas as pd
import numpy as np

user_factor=joblib.load("user_factor.pkl")
movie_factor=joblib.load("movie_factor.pkl")
user_bias=joblib.load("user_bias.pkl")
movie_bias=joblib.load("movie_bias.pkl")
gobel_rating=joblib.load("gobel_rating.pkl")

items=pd.read_csv("ml-100k/u.item", sep="|", encoding="latin-1", header=None, usecols=[0, 1], names=["movie_id", "title"])
titles=dict(zip(items["movie_id"],items["title"]))
train_data = pd.read_csv("ml-100k/ua.base", sep="\t", names=["user_id", "movie_id", "rating", "timestamp"])


st.title("Movie Recommendation System")
st.write("Maximum user id is 943")
num=st.number_input("User ID",min_value=1,max_value=943,value=1,step=1)
movie_prediction=[]
rated=set(train_data[train_data["user_id"]==num]["movie_id"])
if st.button("Recommend"):
    u = num - 1
    for m in range(1682):
        movie_id=m+1
        dot=np.dot(user_factor[u],movie_factor[m])
        pred = gobel_rating + user_bias[u] + movie_bias[m] + dot
        if movie_id not in rated:
            movie_prediction.append((movie_id, pred))
    movie_prediction.sort(key=lambda x: x[1], reverse=True)
    for movie_id,pred in movie_prediction[:10]:
        st.write(titles[movie_id],round(pred,2))
        
