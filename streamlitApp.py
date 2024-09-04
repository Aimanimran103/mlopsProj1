import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
import streamlit as st
#load the model
model=joblib.load("liveModelV1.pk1")

data=pd.read_csv("mobile_price_range_data.csv")
X=data.iloc[:,:-1]
y=data.iloc[:,-1]

#train test split for accuracy calculation only the testing data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#make prediction for x_test
y_pred=model.predict(X_test)

#calculate accuracy
accuracy=accuracy_score(y_test,y_pred)

#Display accuracy
st.title("Model Accuracy and Real-Time Prediction")
st.write(f"Model{accuracy}")

#Real time prediction based on user inputs
st.header("Real-Time Prediction")
input_data=[]
for col in X_test.columns:
    input_value=st.number_input(f'Input for feature {col}',value=0)
    input_data.append(input_value)

#convert input data to dataframe
input_df=pd.DataFrame([input_data],columns=X_test.columns) 

#make prediction
if st.button("Predict"):
    prediction =model.predict(input_df)
    st.write(f'Prediction:{prediction[0]}')