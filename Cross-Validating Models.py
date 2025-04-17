
#welcome to cross varlidation modes
# (    Model Evaluation  )
#this will help us know the quality of model we have created using varius machine learning alog
#so lets  chech how well our model will work in the real world.
#we are going to use real world datasets 
#Create a pipeline that preprocesses the data, trains the model, and checks the quality of our data

#we are going to start by importing the libraries we are going to use

#LETS GO!!!

#we will first import pandas which will help use to read our data wich can be in varius form but we are going
#to use csv file


import pandas as pd

#if you havent installed ensure you have installed those libraries 
#you can use pip install pandas in (windows) or any other tool you are using (colab,ubuntu..etc)
#lets continue loading our sklearn libraries

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold,cross_val_score

df=pd.read_csv("firstdataset2.csv")
#print(df.info())------------this is i was using to get the colums,raw names to ensure accuracy
#the above is used in data cleaning where pandas in essential tool

#we create features
x=df[['math_score']]

#this will be target
y=df['biology_score']


## Create standardizer i have used q but you can replace with any other thing like n,j

q=StandardScaler()

# Create logistic regression object model which we loaded there other like linear,etc
r=LogisticRegression()

# Create a pipeline that standardizes, then runs logistic regression we are utilizing imported tools all
s=make_pipeline(q,r) 

# Create k-Fold cross-validation 

kf=KFold(n_splits=10,shuffle=True,random_state=1)

# Conduct k-fold cross-validation

b=cross_val_score(s,x,y,cv=kf,scoring="accuracy",n_jobs=-1)

# Calculate mean
print('mean is',b.mean())


#............you have now made a cross verlidation model

#.................thank you.................
