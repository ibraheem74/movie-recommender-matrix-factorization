import pandas as pd
import numpy as np
import joblib
df=pd.read_csv("ml-100k/u.data",sep='\t',names=["user_id", "movie_id", "rating","timestamp"])
train_data=pd.read_csv("ml-100k/ua.base",sep="\t",names=["user_id", "movie_id", "rating","timestamp"])
test_data=pd.read_csv("ml-100k/ua.test",sep="\t",names=["user_id", "movie_id", "rating","timestamp"])
user_factor=np.random.rand(df["user_id"].nunique(),20)*0.1
movie_factor=np.random.rand(df["movie_id"].nunique(),20)*0.1
user_bias=np.zeros(df["user_id"].nunique())
movie_bias=np.zeros(df["movie_id"].nunique())
gobel_rating=train_data["rating"].mean()
lr,reg = 0.01,0.02

no_epoch=3
for epoch in range(no_epoch):  
    total =0 
    for index,row in train_data.iterrows():
        u = int(row["user_id"])
        m = int(row["movie_id"])
        r = int(row["rating"])
        u,m=u-1,m-1
        dot=np.dot(user_factor[u],movie_factor[m])
        pred=gobel_rating+user_bias[u]+movie_bias[m]+dot
        old_user = user_factor[u].copy()
        err=r-pred
        total+=err**2

        user_bias[u]+=lr*(err-reg*user_bias[u])
        movie_bias[m]+=lr*(err-reg*movie_bias[m])

        user_factor[u]+=lr*(err*movie_factor[m]-reg*user_factor[u])
        movie_factor[m]+=lr*(err*old_user-reg*movie_factor[m])

        #new_pred=gobel_rating+user_bias[u]+movie_bias[m]+np.dot(user_factor[u], movie_factor[m])
        #new_err=r-new_pred

    print(epoch, total / len(train_data))
    
test_total =0 
for index,row in test_data.iterrows():
    u = int(row["user_id"])
    m = int(row["movie_id"])
    r = int(row["rating"])
    u,m=u-1,m-1
    dot=np.dot(user_factor[u],movie_factor[m])
    pred=gobel_rating+user_bias[u]+movie_bias[m]+dot
    err=r-pred
    test_total+=err**2

print((test_total / len(test_data))**0.5)
user_id=2
u=user_id-1
predictions=[]
for m in range(1682):
    dot=np.dot(user_factor[u],movie_factor[m])
    pred=gobel_rating+user_bias[u]+movie_bias[m]+dot
    predictions.append(pred)
print(predictions[:5])

movie_prediction=[]
rated=set(train_data[train_data["user_id"]==2]["movie_id"])
for m in range(1682):
    movie_id=m+1
    if movie_id not in rated:
        movie_prediction.append((movie_id,predictions[m]))

movie_prediction.sort(key=lambda x: x[1], reverse=True)
print(movie_prediction[:10])

items=pd.read_csv("ml-100k/u.item", sep="|", encoding="latin-1", header=None, usecols=[0, 1], names=["movie_id", "title"])
titles=dict(zip(items["movie_id"],items["title"]))
for movie_id,pred in movie_prediction[:10]:
    print(titles[movie_id],round(pred,2))

joblib.dump(user_factor,"user_factor.pkl")
joblib.dump(movie_factor,"movie_factor.pkl")
joblib.dump(user_bias,"user_bias.pkl")
joblib.dump(movie_bias,"movie_bias.pkl")
joblib.dump(gobel_rating,"gobel_rating.pkl")
print("succesfully saved modul")