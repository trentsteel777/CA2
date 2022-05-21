"""
filename : Farmer's Dashboard.py
This script is the starting point for the execution of this project.

command to run : streamlit run "Farmer_Dashboard.py" --theme.base "light"

This file holds the structure of the entire project and
calls the necessary function as per the requested page.
"""

import streamlit as st
from streamlit_option_menu import option_menu

from utilities import page_title
from visualizations import generate_visuals
from predictions import generate_predictions

st.set_page_config(layout="wide")
st.sidebar.markdown("""<h3 style='text-align: center; color: black;'>{0}</h3>""".format("Farmer's Dashboard"), unsafe_allow_html=True)
st.sidebar.write("")
st.sidebar.write("")

with st.sidebar:
    page = option_menu("", ["Visualization", "Model Prediction"],
                         icons=['graph-up', 'cpu-fill'], orientation='vertical',
                         menu_icon="app-indicator", default_index=0,
                         styles={
                            "container": {"padding": "5!important", "background-color": "#fafafa"},
                            "icon": {"color": "orange", "font-size": "13px"}, 
                            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                            "nav-link-selected": {"background-color": "#F63366"},
                        }
                    )

page_title("Farmer's Dashboard")

if page == "Visualization":
    generate_visuals()

else:
    country = st.sidebar.selectbox("Select country", ["Australia", "Ireland"])
    generate_predictions(country)