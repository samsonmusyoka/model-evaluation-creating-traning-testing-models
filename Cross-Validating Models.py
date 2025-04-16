import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold,cross_val_score

df=pd.read_csv("firstdataset2.csv")
#print(df.info())

x=df[['math_score']]

y=df['biology_score']

q=StandardScaler()
r=LogisticRegression()

s=make_pipeline(q,r)

kf=KFold(n_splits=10,shuffle=True,random_state=1)

b=cross_val_score(s,x,y,cv=kf,scoring="accuracy",n_jobs=-1)

print('mean is',b.mean())
