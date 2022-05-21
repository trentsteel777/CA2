"""
filename : visualizatiions.py

This file contains functions that generates and displays various plots for both countries.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go


from utilities import load_data, transform_data


def data_plotting(data, params, title, y_unit):
    """
    Common Function for generating plot for input parameters

    input ::
        - data : pandas dataframe
        - params : list containing params to display
        - title : plot title text
        - y_unit : y-axis unit 

    output :: 
        - plotly plot
    """

    x = list(data["year"])

    fig = px.line(data, x='year', y=params, markers=True)
    fig.update_layout(title_text=title,
                        yaxis_title= y_unit,
                        xaxis = dict(tickmode = 'array',tickvals = x,ticktext = x))

    st.plotly_chart(fig,use_container_width=True)

    st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)


def generate_visuals():
    """
    Main Function for generating visuals for the selected country

    input ::

    output :: 
        - plotly plots
    """

    aus_data, ire_data = load_data()

    # selected params for both countries
    aus_params = ['beef_per_kg', 'wheat_production', 'audusd', 'milk_production', 'diesel', 'swine_production_x']
    ire_params = ['year', 'target_cattle', 'cereals', 'eurusd', 'milk', 'diesel', 'pigs'] 

    aus_data = aus_data[aus_params]
    ire_data = ire_data[ire_params]

    # columns renaming for better understanding
    aus_data = aus_data.rename(columns={'diesel': 'Australia_diesel', 'milk_production': 'Australia_milk'})
    ire_data = ire_data.rename(columns={'diesel': 'Ireland_diesel', 'milk': 'Ireland_milk'})

    # transforming data
    aus_data = transform_data(data=aus_data, params=['wheat_production', 'Australia_milk', 'swine_production_x'])
    ire_data = transform_data(data=ire_data, params=['cereals', 'Ireland_milk', 'pigs'])

    # combined data of both country to further use
    df = pd.concat([ire_data, aus_data], axis=1)

    st.sidebar.write("")
    st.sidebar.write("")

    # year selection slider on sidebar
    start_year, end_year = st.sidebar.select_slider('Select year range',options=range(1999, 2023),value=(2003, 2015))

    # filtering data based on selected year
    df = df[(df['year']>= start_year) & (df['year'] <= end_year)]

    x = list(df["year"])

    ####################################### USD conversion plot #########################################

    st.markdown("""<br><h3 style='text-align: left; color: black;'>USD Conversion</h3>""", unsafe_allow_html=True)
    data_plotting(data=df, 
                    params=["audusd", "eurusd"], 
                    title="USD Conversion Rate of Australian dollars and Euro between between <b>"+str(start_year)+"</b> and <b>"+str(end_year)+"</b>", 
                    y_unit="Conversion Rate")

    #################################### Diesel price plot #############################################

    st.markdown("""<br><h3 style='text-align: left; color: black;'>Diesel price</h3>""", unsafe_allow_html=True)
    data_plotting(data=df, 
                    params=["Australia_diesel", "Ireland_diesel"], 
                    title="Diesel price in Australia and Ireland from <b>"+str(start_year)+"</b> to <b>"+str(end_year)+"</b>", 
                    y_unit="price")

    # side by side plot
    col1, col2 = st.columns(2)

    #################################### Beef price plot #############################################
    with col1:

        st.markdown("""<br><h3 style='text-align: left; color: black;'>Beef price</h3>""", unsafe_allow_html=True)
        
        fig = px.bar(df, x="year", y="beef_per_kg", color= "beef_per_kg"  , barmode = 'stack')
        
        fig.update_layout(xaxis = dict(tickmode = 'array',tickvals = x,ticktext = x), 
                        title_text="Beef per KG price in Australia from <b>"+str(start_year)+"</b> to <b>"+str(end_year)+"</b>",
                        yaxis_title="price")
        
        st.plotly_chart(fig,use_container_width=True)

    #################################### Cattle plot ################################################
    with col2:

        st.markdown("""<br><h3 style='text-align: left; color: black;'>Cattle price</h3>""", unsafe_allow_html=True)

        fig = px.bar(df, x="year", y="target_cattle", color= "target_cattle"  , barmode = 'stack')
        fig.update_layout(xaxis = dict(tickmode = 'array',tickvals = x,ticktext = x),
                        title_text="Cattle price in Ireland from <b>"+str(start_year)+"</b> to <b>"+str(end_year)+"</b>",
                        yaxis_title="price")

        st.plotly_chart(fig,use_container_width=True)

    #################################### Milk plot (Bar plot) #############################################

    st.markdown("""<br><h3 style='text-align: left; color: black;'>Milk Production (Normalized)</h3>""", unsafe_allow_html=True)

    y1 = list(df["Australia_milk"])
    y2 = list(df["Ireland_milk"])

    annotations1 = [dict(
            x=xi-0.18,
            y=yi,
            text=str(yi),
            xanchor='auto',
            yanchor='bottom',
            showarrow=False,
        ) for xi, yi in zip(x, y1)]
    
    annotations2 = [dict(
                x=xi+0.18,
                y=yi,
                text=str(yi),
                xanchor='auto',
                yanchor='bottom',
                showarrow=False,
            ) for xi, yi in zip(x, y2)]
    
    annotations = annotations1 + annotations2

    trace1 = go.Bar(x=x,y=y1,textposition='auto',name='Australia_milk')

    trace2 = go.Bar(x=x,y=y2,textposition='auto',name='Ireland_milk')

    data = [trace1, trace2]

    layout = go.Layout(barmode='group',annotations=annotations)

    fig = go.Figure(data=data, layout=layout)
    
    fig.update_layout(xaxis = dict(tickmode = 'array',tickvals = x,ticktext = x))
    fig.update_layout(title_text="Milk Production in Australia and Ireland from <b>"+str(start_year)+"</b> to <b>"+str(end_year)+"</b>",
                    yaxis_title= "Percentage change", xaxis_title="year")

    st.plotly_chart(fig,use_container_width=True)

    #################################### Wheat & Cereal plot #############################################

    st.markdown("""<br><h3 style='text-align: left; color: black;'> Wheat vs Cereal (Normalized)</h3>""", unsafe_allow_html=True)
    data_plotting(data=df, 
                    params=["wheat_production", "cereals"], 
                    title="Wheat Production in Australia and Cereal price in Ireland from <b>"+str(start_year)+"</b> to <b>"+str(end_year)+"</b>", 
                    y_unit="Percentage change")

    #################################### Swine & Pig plot #############################################

    st.markdown("""<br><h3 style='text-align: left; color: black;'>Swine vs Pig production (Normalized)</h3>""", unsafe_allow_html=True)

    y1 = list(df["swine_production_x"])
    y2 = list(df["pigs"])

    trace1 = go.Bar(name = 'swine production',x = list(df["year"]),y = list(df["swine_production_x"]))

    trace2 = go.Bar(name = 'pigs production',x = list(df["year"]),y = list(df["pigs"]))

    plot = go.Figure(data=[trace1, trace2])
    
    plot.update_layout(barmode='stack',
                        xaxis = dict(tickmode = 'array',tickvals = x,ticktext = x),
                        yaxis_title="Percentage change", xaxis_title="year",
                        title_text="Swine Production in Australia and Pigs price in Ireland from <b>"+str(start_year)+"</b> to <b>"+str(end_year)+"</b>")
    st.plotly_chart(plot,use_container_width=True)