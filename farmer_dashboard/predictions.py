"""
filename : predictions.py

This file contains functions for Predictions using the trained model.
"""

import streamlit as st
from xgboost import XGBRegressor

from utilities import load_data, load_selected_features

def load_model(country_name):
    """
    Function for model loading.

    input ::
        - country_name : Australia / Ireland 

    output :: 
        - model
    """
    model = XGBRegressor()

    if country_name=="Australia":
        model.load_model("assets/model_australian_beef.json")
    else:
         model.load_model("assets/model_ireland_cattle.json")

    return model

@st.cache
def generate_data(input_values, country_name):
    """
    Function for generating data point for model prediction.

    input ::
        - input_values : input taken from user from UI
        - country_name : Australia / Ireland 

    output :: 
        - dataframe containing single data point
    """

    aus_df, ire_df = load_data()

    if country_name=="Australia":
        features = load_selected_features(country_name)
        aus_df = aus_df[features]
        df = aus_df.sample()
        df[['audusd', 'milk_production', 'diesel', 'swine_production_x']] = input_values

    else:
        features= load_selected_features(country_name)
        ire_df = ire_df[features]
        df = ire_df.sample()
        df[['cereals', 'eurusd', 'milk', 'diesel', 'pigs']] = input_values

    return df


def generate_predictions(country_name):
    """
    Function for model predictions.

    input ::
        - country_name : Australia / Ireland 

    output :: 
        - predicted value
    """
    if country_name=="Ireland":

        st.markdown("""<br><h3 style='text-align: left; color: black;'>Cattle Price prediction</h3>""", unsafe_allow_html=True)

        with st.form(key='columns_in_form'):
            cols = st.columns(5)
            with cols[0]:
                cereals = cols[0].number_input('Cereal price', value =100, min_value =50, max_value=200)
            
            with cols[1]:
                eurusd = cols[1].number_input('USD conversion rate', value =1.1, step=0.1, min_value =0.1, max_value=10.0)

            with cols[2]:
                milk = cols[2].number_input('Milk price', value =110, min_value =50, max_value=175)
            
            with cols[3]:
                diesel = cols[3].number_input('Diesel price', value =125, min_value =50, max_value=200)

            with cols[4]:
                pigs = cols[4].number_input('Pig price', value =100, min_value =50, max_value=125)
            
            submitted = st.form_submit_button('Submit')

        if submitted:

            model = load_model(country_name)
            data = generate_data([cereals, eurusd, milk, diesel, pigs], country_name)
            y_pred_xgb = model.predict(data)[0]

            st.markdown("""<br><h4 style='text-align: left; color: black;'>Cattle Price predicted : {0}</h4>""".format(str(y_pred_xgb)[:6]), unsafe_allow_html=True)


    if country_name=="Australia":

        st.markdown("""<br><h3 style='text-align: left; color: black;'>Beef Price prediction</h3>""", unsafe_allow_html=True)

        with st.form(key='columns_in_form'):
            cols = st.columns(4)
            
            with cols[0]:
                audusd = cols[0].number_input('USD conversion rate', value =0.6, step=0.1, min_value =0.1, max_value=10.0)

            with cols[1]:
                milk_production = cols[1].number_input('Milk production', value =750, min_value =700, max_value=1000)
            
            with cols[2]:
                diesel = cols[2].number_input('Diesel price', value =125, min_value =50, max_value=200)

            with cols[3]:
                swine_production_x = cols[3].number_input('Swine production', value =35.13, step =0.1, min_value =25.1, max_value=40.0)
            
            submitted = st.form_submit_button('Submit')

        if submitted:

                model = load_model(country_name)
                data = generate_data([audusd, milk_production, diesel, swine_production_x], country_name)
                y_pred_xgb_2 = model.predict(data)[0]
                st.markdown("""<br><h4 style='text-align: left; color: black;'>Beef Price predicted : {0}</h4>""".format(str(y_pred_xgb_2)[:4]), unsafe_allow_html=True)