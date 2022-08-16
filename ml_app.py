# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

def load_file(filename):
  with open(filename+'.pickle', 'rb') as handle:
     return pickle.load(handle)

st.title('Predicting whether customer will get checked in ')
age=st.number_input('Enter age of customer')
nationality=st.text_input("Enter nationality of customer","")
dist_ch=st.text_input("Enter distribution channel of customer")
days_creat=st.number_input("Enter days since creation for customer")
avg_lead=st.number_input("Enter average lead time of customer")
room_night=st.number_input("Enter roomnights of customer")
person_night=st.number_input("Enter personnights of customer")
lodging_rev=st.number_input("Enter lodging revenue for customer")
mrkt_seg=st.text_input("Enter market_segment of customer")
other_rev=st.number_input("Enter other revenue for customer")
days_last=st.number_input("Enter DaysSinceLastStay for customer")

top_features=['RoomNights', 'DaysSinceLastStay', 'OtherRevenue', 'AverageLeadTime',
       'LodgingRevenue', 'PersonsNights', 'Age', 'DaysSinceCreation',
       'MarketSegment_6', 'DistributionChannel_2', 'Nationality_27',
       'DaysSinceCreation_sin', 'DaysSinceCreation_cos', 'Nationality_2',
       'DistributionChannel_1', 'MarketSegment_1', 'Nationality_6',
       'Nationality_14', 'Nationality_11']

my_dict={
    'Age':age, 
    'Nationality':nationality, 
    'DistributionChannel':dist_ch,
    'DaysSinceCreation':days_creat, 
    'RoomNights':room_night,
    'PersonsNights':person_night, 
    'LodgingRevenue':lodging_rev, 
    'MarketSegment':mrkt_seg, 
    'AverageLeadTime':avg_lead,
    'OtherRevenue':other_rev, 
    'DaysSinceLastStay':days_last
    
    }
scaler=load_file('scaler_final')
ohe=load_file('ohe_final')
cyclical=load_file('cylical_final')
model=load_model('model.h5')
def predict(test):
    test_1=ohe.transform(test)
    test_2=cyclical.transform(test_1)
    train=test_2[top_features]
    train_scaled=scaler.transform(train)
    pred=model.predict(train_scaled)
    if pred[0][0]>=0.5:
        result='CHECKED IN'
    else:
        result='NOT CHECKED IN'
    return result
if st.button("Predict"):
    test=pd.DataFrame([my_dict], columns=my_dict.keys())
    st.write(predict(test))
