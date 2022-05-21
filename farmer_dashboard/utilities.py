"""
filename : utilities.py

This file contains common functions which are being called throughout the app.
"""

import pandas as pd
import streamlit as st

def page_title(page):
	"""function to display page title on every page"""

	st.markdown("""<h2 style='text-align: center; color: black;'>{0}</h2>""".format(page), unsafe_allow_html=True)
	st.write("")


@st.cache
def process_data(df):
    """
    Function for pre-processing data

    input ::
        - pandas dataframe

    output :: 
        - processed pandas dataframe
    """

    df = df.rename(columns={'Unnamed: 0': 'Timestamp'})
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['year'] = df['Timestamp'].dt.year
    df.index = df['Timestamp']
    df = df.resample('Y').mean()
    df = df.round(2)

    return df


@st.cache
def load_data():
    """
    Function for loading data

    input ::

    output :: 
        - pandas dataframe containing data of Austrailia and Ireland
    """
    
    aus_filename = "df_final_australian_beef_feat_cols_and_target_values_1999-2020.csv"
    ire_filename = "df_final_irish_beef_feat_cols_and_target_values_1999-2022.csv"

    aus_df = pd.read_csv("assets/"+aus_filename)
    ire_df = pd.read_csv("assets/"+ire_filename)

    aus_df = process_data(aus_df)
    ire_df = process_data(ire_df)

    return aus_df, ire_df


@st.cache
def transform_data(data, params):
    """
    Function for transforming data in 0 to 100

    input ::
        - data : pandas dataframe
        - params : params to normalize

    output :: 
        - normalized dataframe
    """

    df = data[params]
    df = (df-df.min())/(df.max()-df.min())
    df = df.round(2)
    data[params] = df

    return data
    
def load_selected_features(country_name):
    """
    Function for loading features used in model training.

    input ::
        - country_name : Australia / Ireland 

    output :: 
        - feature list
    """
    if country_name=="Australia":
        features = ['cpi_all','milk_powder_production','poultry_production','cpi_residential','cpi_all_ex_food_energy','pbeefusdm','cpi_prod_index',
                    'meat_exports','meat_imports','cpi_food','butter_production','beef_and_veal_production','audusd','nonfatmilk_production',
                    'audcny','meat_supply','swine_production_x','audnzd','meat_distribution','meat_consumption','milk_production',
                    'crude_price','barley_production','millet_production','petrol','diesel']
    else:
         features=['heifers_300-349kg','bullocks_450-499kg','bullocks_400-449kg','heifers_400-449kg','bullocks_350-399kg','bullocks_250-299kg',
                'bullocks_500-549kg','heifers_200-249kg','heifers_350-399kg','heifers_250-299kg','other_goods_and_services','compound_feeding_stuffs_for_poultry',
                'bullocks_300-349kg','food_and_non-alcoholic_beverages','straight_fertilisers','pork_loin_chops_per_kg','veterinary_expenses',
                'brandy_take_home_70cl_bottle','potatoes','plant_protection_products','jam_per_lb','sirloin_steak_per_kg','marmalade_per_lb',
                'poultry','sheep','best_back_rashers_per_kg','eurusd','pk_fertilisers','bullocks_200-249kg','compound_feeding_stuffs_for_pigs',
                'cents_per_kg','brent_price','irish_cheddar_per_kg','eurgbp','rain','cooked_ham_per_kg','milk','motor_fuel','petrol_unleaded_per_ltr',
                'pork_steak_per_kg','feeding_stuffs','crop_output','vodka_take_home_70cl_bottle','temp','pork_sausages_per_kg','sherry_take_home_75cl_bottle',
                'vegetables','fertilisers','ham_fillet_per_kg','onions_per_kg','pigs','potatoes_10_kg','sunshine','ale_draught_bar_1_pint','lamb_loin_chops_per_kg',
                'crude_price','whiskey_take_home_70cl_bottle','bullocks_550kg+','electricity','mushrooms_per_kg','diesel','carrots_per_kg','butter_per_lb',
                'lambs_liver_per_kg','lamb_gigot_chops_per_kg','pbeefusdm','cereals','tomatoes_per_kg','compound_feeding_stuffs_for_cattle_excluding_calves',
                'stout_draught_bar_1_pint','diesel_per_ltr','broccoli_per_kg','all_items','cattle','petrol','npk_fertilisers']
    
    return features