import streamlit as st
import re
import sqlalchemy
from sqlalchemy import create_engine
import warnings
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import explained_variance_score, r2_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error as MSE
from pathlib import Path
import xgboost as xg 
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import datetime
import os
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import matplotlib.pyplot as plt
import joblib as jlib

engine = create_engine("sqlite:///Resources/insurance.sqlite")
warnings.filterwarnings('ignore')
conn = engine.raw_connection()
df = pd.read_sql_query(sql="SELECT age, sex, bmi, children, smoker, region, ROUND(charges, 2) AS charges FROM insurance", con=conn)
df1 = df.copy()

col_to_encode = ['sex', 'smoker', 'region']

X_OneHot_model = OneHotEncoder(sparse_output=False)
X_OH_fit = X_OneHot_model.fit(df[col_to_encode])

df_encoded =  pd.DataFrame(X_OneHot_model.transform(df[col_to_encode]), columns=X_OneHot_model.get_feature_names_out())
df1 = df1.drop(col_to_encode, axis=1).merge(df_encoded, left_index=True, right_index=True)

X_min_max_scaler = MinMaxScaler()
X_fit_scaler = X_min_max_scaler.fit(df1[['age','bmi','children']])
df1[['age','bmi','children']] = X_fit_scaler.transform(df1[['age','bmi','children']])
X = df1.drop('charges', axis = 1)
y_min_max_scaler = MinMaxScaler()
y_fit_scaler = y_min_max_scaler.fit(df1[['charges']])
y = y_fit_scaler.transform(df1[['charges']])
y.ravel().reshape(1,-1)
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state = 123)
X_min_max_model = jlib.load('train model/X_min_max_fitb')
y_min_max_model = jlib.load('train model/y_min_max_fitb')
X_oneHot_model = jlib.load('train model/oneHotModelb')
pred_model = jlib.load('train model/xgboost_modelinsurancea')


im = Image.open("images/tab_icon.png")
st.set_page_config(page_title="Medical Insurace Prediction", layout="wide", page_icon=im)
left, center, right = st.columns([2,5,1.75])
with left:
    st.header('')
    st.image('images/medical-center.png', width=250)

with center:
    st.header("")
    st.header("")
    st.header("")
    st.markdown("<h1 style='text-align:center;color:#E2A3F3'>MEDICAL INSURANCE PREDICTIONS</h1>", unsafe_allow_html=True)

with right:
    st.image('images/healthcare.png', width=250)

left,center,right = st.columns([1, 5, 1])
with left:
    visual = st.radio('**INFORMATION**', options=['**ABOUT**','**DATA**','**ML MODEL**','**PREDICTION FORM**', '**CONCLUSION**'], index=0)
with center:
    if visual == '**ABOUT**':
        st.markdown("<h1 style='text-align:center'>ARE YOU AT RISK FOR A HIGHER MEDICAL INSURANCE PREMIUM?</h1>", unsafe_allow_html=True)
        left,center,right = st.columns([2,2,2])
        with left:
            st.write("")
        with center:
            st.image("images/financial-profit.png", use_column_width=True)
        with right:
            st.write("")
        st.header("Medical insurance premiums take a pretty heavy chunk of our yearly earnings! The most common factors that insurance companies will use to dictate the cost of your premium are your age, BMI (body mass index), gender, smoking status, number of children, and where you live. This Machine Learning Model will give you the expected premium that insurance companies should quote you based off of your information.")

    if visual == '**DATA**':

        st.markdown("<h2 style='text-align:center'>MEDICAL INSURANCE DATASET</h2>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

        st.title("")
        st.markdown("<h2 style='text-align:center'>SMOKER VS NON SMOKER</h2>", unsafe_allow_html=True)
        st.bar_chart(df['smoker'].value_counts())

        st.title("")
        st.markdown("<h2 style='text-align:center'>REGION</h2>", unsafe_allow_html=True)
        st.bar_chart(df['region'].value_counts())

        st.title("")
        st.markdown("<h2 style='text-align:center'>FREQUENCY OF CHARGES</h2>", unsafe_allow_html=True)
        fig = plt.figure(figsize=(7,2))
        ax = plt.axes()
        ax.hist(df['charges'], bins=20)
        plt.xlabel('charges')
        plt.ylabel('Frequency')
        st.pyplot(fig)

        st.title("")
        st.markdown("<h2 style='text-align:center'>FREQUENCY OF BMI</h2>", unsafe_allow_html=True)
        fig = plt.figure(figsize=(7,2))
        ax = plt.axes()
        ax.hist(df['bmi'], bins=20)
        plt.xlabel('BMI')
        plt.ylabel('Frequency')
        st.pyplot(fig)


    if visual == '**ML MODEL**':
        st.image('images/ml model.png')
        st.header("XGBoost is one of the most popular supervised learning machine models. This extreme grade boosting machine uses the decision tree ensemble, training each subset to each tree to combine  to a final prediction model. As each model in succession correct the errors of the previous on a graded scale, this allows XGboost machine learning to give you the best linear regression for your data set with lower risk of overfitting.")
        st.header("For our machine model, XGboost loops through multiple settings of 6 different parameters setting to perform almost 13,000 fittings to find combination of parameters for the best accurate machine. With the best fitted parameters shown below, our machine model was able to predict 83% of testing data and 89 % of training data, with RMSE of 0.06239, all for under 10 minutes.")
        left,center,right = st.columns([1,3,1]) 
        with left:
            st.write("")
        with center:
            st.image('images/Feature.png', use_column_width=True)
        with right:
            st.write("")
    if visual == '**PREDICTION FORM**':

        st.markdown("<h2 style='text-align:center'>MEDICAL INSURANCE PREDICTING FORM</h2>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:center'>Please answer the following questions to get your medical insurance prediction: </h3>", unsafe_allow_html=True)

        st.markdown("<h4>How old are you?</h4>", unsafe_allow_html=True)
        age = st.text_input(label="age", label_visibility="collapsed",placeholder="Please enter your age")
        if re.search('[a-zA-Z]', age):
            st.error('ERROR: Enter numerical value!', icon="🚨")
        elif '.' in age:
            st.error('ERROR: Enter whole number!', icon="🚨")
        elif re.search('[\D]', age):
            st.error('ERROR: Enter numerical value!', icon="🚨")

        st.markdown("<h4>What is your BMI (Body Mass Index)?</h4>", unsafe_allow_html=True)
        bmi = st.text_input(label="bmi", label_visibility="collapsed",placeholder="Please enter your BMI")
        if re.search('[a-zA-Z]', bmi) or '.' == bmi or '..' in bmi:
            st.error('ERROR: Enter numerical value!', icon="🚨")
        elif '.' in bmi:
            pass
        elif re.search('[\D]', bmi):
            st.error('ERROR: Enter numerical value!', icon="🚨")

        st.markdown("<h4>How many children do you have?</h4>", unsafe_allow_html=True)
        children = st.text_input(label="children", label_visibility="collapsed",placeholder="Please enter the number of children you have")
        if re.search('[a-zA-Z]', children):
            st.error('ERROR: Enter numerical value!', icon="🚨")
        elif '.' in children:
            st.error('ERROR: Enter whole number!', icon="🚨")
        elif re.search('[\D]', children):
            st.error('ERROR: Enter numerical value!', icon="🚨")

        st.markdown("<h4>What is your gender?</h4>", unsafe_allow_html=True)
        gender = st.radio(label='gender', label_visibility="collapsed", options=['**female**', '**male**'], index=None)

        st.markdown("<h4>Do you smoke?</h4>", unsafe_allow_html=True)
        smoker = st.radio(label='smoker', label_visibility="collapsed", options=['**yes**', '**no**'], index=None)

        st.markdown("<h4>Which US region do you live in?</h4>", unsafe_allow_html=True)
        region = st.radio(label='region', label_visibility="collapsed", options=['**northeast**', '**northwest**', '**southeast**', '**southwest**'], index=None)

        output_cost = st.button("PREDICT COST", type="primary")
        if output_cost and (age == '' or gender == None or bmi == '' or children == '' or smoker == None or region == None):
            st.error('ERROR: Please answer all the above fields to get your medical insurance prediction!', icon="🚨")
        predict = []
        if age != '' and gender != None and bmi != '' and children != '' and smoker != None and region != None:
            predict.extend([int(age), float(bmi), int(children), str(gender).replace('*', ''), str(smoker).replace('*', ''), str(region).replace('*', '')])
            if output_cost:
                def predict_ins_charges(X_raw_input_arr):

                    feature_cols = ['age', 'bmi', 'children', 'sex', 'smoker','region']
                    input_df = pd.DataFrame([dict(zip(feature_cols, X_raw_input_arr))])

                    cols_to_encode = ['sex', 'smoker', 'region']
                    one_hot_encoded_df =  pd.DataFrame(X_oneHot_model.transform(input_df[cols_to_encode]), columns=X_oneHot_model.get_feature_names_out())

                    col_to_scale = ['age', 'bmi', 'children']
                    input_df[col_to_scale] = X_min_max_model.transform(input_df[col_to_scale])


                    input_df = input_df.drop(cols_to_encode, axis = 1).merge(one_hot_encoded_df, left_index=True, right_index=True)

                    y_pred_scaled = pred_model.predict(input_df)
                    y_pred_scaled = np.reshape(y_pred_scaled, (-1, 1))
    
                    ISR = y_min_max_model.inverse_transform(y_pred_scaled).ravel()[0]
                    return ISR
                cost = predict_ins_charges(predict)
                quote = "Your Yearly Medical Insurance Premium Prediction: ${:,.2f}".format(cost)
                st.markdown(f"<h1>{quote}</h1>", unsafe_allow_html=True)
    if visual == '**CONCLUSION**':
        left,center,right = st.columns([2,2,2])
        with left:
            st.write("")
        with center:
            st.image('images/conclusion.jpg', use_column_width=True)
        with right:
            st.write("")
        st.header("Through this dataset we are able to predict the cost of an individual's medical insurance based on their life style situation. We used Xgboost, Scikit-learn, and other libraries to show the charge of insurance per person could be affected by their age, gender, region and other factors.")

with right:
    st.write("")
        