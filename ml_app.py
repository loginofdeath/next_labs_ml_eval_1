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
srking=st.text_input("Enter SRKingBed for customer")
dist_ch=st.text_input("Enter disctribution channel of customer")
days_creat=st.number_input("Enter days since creation for customer")
avg_lead=st.number_input("Enter average lead time of customer")
room_night=st.number_input("Enter roomnights of customer")
person_night=st.number_input("Enter personnights of customer")
days_first=st.number_input("Enter DaysSinceFirstStay for customer")
lodging_rev=st.number_input("Enter lodging revenue for customer")
mrkt_seg=st.text_input("Enter market_segment of customer")
other_rev=st.number_input("Enter other revenue for customer")
days_last=st.number_input("Enter DaysSinceLastStay for customer")

top_features=['DaysSinceFirstStay',
 'RoomNights',
 'PersonsNights',
 'DaysSinceLastStay',
 'OtherRevenue',
 'LodgingRevenue',
 'AverageLeadTime',
 'Age',
 'MarketSegment_2',
 'DistributionChannel_2',
 'DaysSinceCreation',
 'DaysSinceCreation_cos',
 'DaysSinceCreation_sin',
 'MarketSegment_3',
 'DistributionChannel_1',
 'SRKingSizeBed',
 'MarketSegment_6',
 'DistributionChannel_3',
 'Nationality_18',
 'Nationality_48',
 'Nationality_2']

my_dict={
    'Age':age, 
    'Nationality':nationality, 
    'SRKingSizeBed':srking, 
    'DistributionChannel':dist_ch,
    'DaysSinceFirstStay':days_first, 
    'DaysSinceCreation':days_creat, 
    'RoomNights':room_night,
    'PersonsNights':person_night, 
    'LodgingRevenue':lodging_rev, 
    'MarketSegment':mrkt_seg, 
    'AverageLeadTime':avg_lead,
    'OtherRevenue':other_rev, 
    'DaysSinceLastStay':days_last
    
    }
scaler=load_file('C:\\Users\\soura\\.spyder-py3\\scaler_final')
ohe=load_file('C:\\Users\\soura\\.spyder-py3\\ohe_final')
cyclical=load_file('C:\\Users\\soura\\.spyder-py3\\cylical_final')
model=load_model('C:\\Users\\soura\\.spyder-py3\\model.h5')
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
