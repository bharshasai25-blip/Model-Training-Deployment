from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium
from folium.plugins import MarkerCluster
import streamlit as st
import os
import streamlit.components.v1 as components
import joblib
from pathlib import Path

# Define base directory
BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Data Visualization App", layout="wide")
st.sidebar.title("Navigation")
dataset_selection = st.sidebar.selectbox("Select Dataset", options=["FBI Crime Data", "LightGBM Crime Forecast", "XGBoost Crime Forecast", "SARIMAX Crime Forecast"])

# Building Map visualization with Folium and Streamlit
@st.cache_resource
def create_crime_map(crime_location_df):

     crime_map = folium.Map(
        location=[
            crime_location_df['Latitude'].mean(),
            crime_location_df['Longitude'].mean()
        ],
        zoom_start=12
    )

     marker_cluster = MarkerCluster().add_to(crime_map)

     for row in crime_location_df.itertuples():  # Faster than iterrows()
        folium.CircleMarker(
            location=[row.Latitude, row.Longitude],
            radius=max(row.Crime_Count * 0.001, 2),  # Prevent tiny circles
            popup=f"{row.NEIGHBOURHOOD} - {row.HUNDRED_BLOCK}: {row.Crime_Count} crimes",
            color='blue',
            fill=True,
            fill_color='blue'
        ).add_to(marker_cluster)

     return crime_map

# Data
# Conditional logic to display the appropriate dataset and visualizations based on user selection
if dataset_selection == "FBI Crime Data":
    DATA_DIC = BASE_DIR / "VS Code Visualization Datasets"
    data_path1 = DATA_DIC / "FBI_Crime_df.csv"
    df = pd.read_csv(data_path1)
    df['YEAR'] = df['YEAR'].astype(int)
    df['MONTH'] = df['MONTH'].astype(int)

    st.title("FBI Crime Data Trends and Visuals App")
    st.write("This app demonstrates various data visualization techniques using different libraries.")
    st.write("Running file:", os.path.abspath(__file__))
    st.write("Data Preview:")
    st.dataframe(df.head(10))
    st.sidebar.header("Visualization Options")
    st.sidebar.subheader("Select Visualizations to Display")
    visual_selection = st.sidebar.selectbox("Choose Visualization Type",
                         options=["Show All Visualizations", "Yearly Crime Trend", "Yearly Crime Trend by Type", "Monthly Crime Trend", "Monthly Crime Trend by Type", "Crime Location Map", "Neighbourhoods Vs Crime Count",
                                  "Neighbourhoods Vs Crime Type Heatmap", "Neighbourhoods Crime Counts Yearly", "Neighbourhoods Crime Counts Yearly Trend", "Neighbourhoods Crime Type Counts Yearly", "Neighbourhoods Crime Types Yearly Trend",
                                  "Neighbourhoods Crime Counts Monthly", "Neighbourhoods Crime Counts Monthly Trend", "Neighbourhoods Crime Type Counts Monthly", "Neighbourhoods Crime Type Counts Monthly Trend"], key="viz_type")
    if visual_selection == "Show All Visualizations":
       st.write("All visualizations are displayed below:")
    # Yearly Crime Trend Line Plot
       yearly_crime_count = df.groupby('YEAR')['TYPE'].count()
       st.subheader("Yearly Crime Trend")
       fig = px.line(x=yearly_crime_count.index, y=yearly_crime_count.values,
                  labels={'x': 'Year', 'y': 'Number of Crimes'},
                  title='Yearly Crime Trend')
       st.plotly_chart(fig)

    # Yearly Crime Trend Bar Plot
       yearly_crime_count_per_type = df.groupby(['YEAR', 'TYPE']).size().reset_index().rename(columns={0: 'Count'})
       st.subheader("Yearly Crime Count by Type")
       fig = px.bar(yearly_crime_count_per_type, x='YEAR',
                 y='Count', color='TYPE', hover_data= ['TYPE', 'YEAR', 'Count'],
                 labels={'Count': 'Number of Crimes', 'YEAR': 'Year'},
                 title='Yearly Crime Trend by Type')
       st.plotly_chart(fig)

    # Monthly Crime Trend Line Plot
       monthly_crime_count = df.groupby(['YEAR', 'MONTH']).size().reset_index().rename(columns={0: 'Count'})
       monthly_crime_count_df = monthly_crime_count.copy()
       monthly_crime_count_df['Month_name'] = monthly_crime_count_df['MONTH'].apply(lambda x: pd.to_datetime(str(x), format='%m').strftime('%B'))
       st.subheader("Monthly Crime Trend")
       fig = px.line(monthly_crime_count_df, x='Month_name', y='Count', color='YEAR',
                  labels={'Month_name': 'Month', 'Count': 'Number of Crimes', 'YEAR': 'Year'},
                  title='Monthly Crime Trend')
       st.plotly_chart(fig)

    # Monthly Crime Trend Bar Plot
       monthly_crime_count_per_type = df.groupby(['TYPE', 'YEAR', 'MONTH']).size().reset_index().rename(columns={0: 'Count'})
       monthly_crime_count_per_type_df = monthly_crime_count_per_type.copy()
       monthly_crime_count_per_type_df['Month_name'] = monthly_crime_count_per_type_df['MONTH'].apply(lambda x: pd.to_datetime(str(x), format='%m').strftime('%B'))
       st.subheader("Monthly Crime Count by Type")
       fig = px.bar(monthly_crime_count_per_type_df, x='Month_name', y='Count', color='TYPE', facet_col='YEAR', facet_col_wrap=4,
                 hover_data= ['TYPE', 'MONTH', 'Count'],
                 labels={'Count': 'Number of Crimes', 'Month_name': 'Month'},
                 title='Monthly Crime Trend by Type')
       fig.update_layout(height=800, width=1200)
       st.plotly_chart(fig)

    # Crime Location Map
       neighbourhood_crime_count = df.groupby(['NEIGHBOURHOOD'])['TYPE'].count().reset_index().rename(columns={'TYPE': 'Crime_Count'})
       latitude_longitude_df = df.groupby(['NEIGHBOURHOOD'])[['Latitude', 'Longitude']].mean().reset_index()
       crime_location_df = pd.merge(neighbourhood_crime_count, latitude_longitude_df, on=['NEIGHBOURHOOD'], how='inner')
       crime_map = folium.Map(location=[crime_location_df['Latitude'].mean(), crime_location_df['Longitude'].mean()], zoom_start=12)
       for _, row in crime_location_df.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=row['Crime_Count'] * 0.001,
            popup=f"{row['NEIGHBOURHOOD']}: {row['Crime_Count']} crimes",
            color='blue',
            fill=True,
            fill_color='blue'
        ).add_to(crime_map)
       st.subheader("Crime Location Map")
       st_folium = st.components.v1.html(crime_map._repr_html_(), width=1000, height=500)

    # Neighbourhoods Vs Crime Count Bar Plot
       neighbourhood_crime_totals = df['NEIGHBOURHOOD'].value_counts().reset_index()
       neighbourhood_crime_totals.columns = ['NEIGHBOURHOOD', 'Crime_Count']
       st.subheader("Neighbourhoods Vs Crime Count")
       fig = px.bar(neighbourhood_crime_totals, x='NEIGHBOURHOOD', y='Crime_Count',
                 labels={'NEIGHBOURHOOD': 'Neighbourhood', 'Crime_Count': 'Number of Crimes'},
                 title='Neighbourhoods Vs Crime Count')
       st.plotly_chart(fig)

    # Neighbourhoods Vs Crime Type Heatmap
       neighbourhood_crime_type = df.groupby(['NEIGHBOURHOOD', 'TYPE']).size().reset_index().rename(columns={0: 'Crime_Count'})
       st.subheader("Neighbourhoods Vs Crime Type Heatmap")
       fig = px.density_heatmap(neighbourhood_crime_type, x='TYPE', y='NEIGHBOURHOOD', z='Crime_Count',
                    labels={'x': 'Crime Type', 'y': 'Neighbourhood', 'color': 'Number of Crimes'},
                    title='Neighbourhoods Vs Crime Type Heatmap')
    # Increase the figure size for better visibility
       fig.update_layout(width=1000, height=800)
    # Render with fixed width (not auto-fill) so the explicit size is used
       st.plotly_chart(fig, width='content')

    # Neighbourhoods Crime Count Yearly Bar Plot
       neighbourhood_crime_count_yearly = df.groupby(['NEIGHBOURHOOD', 'YEAR']).size().reset_index().rename(columns={0: 'Crime_Count'})
       st.subheader("Neighbourhoods Crime Count Yearly")
       fig = px.bar(neighbourhood_crime_count_yearly, x='NEIGHBOURHOOD', y='Crime_Count', color='YEAR',
                 hover_data= {'YEAR': True, 'Crime_Count': True},
                 labels={'Crime_Count': 'Number of Crimes', 'NEIGHBOURHOOD': 'Neighbourhood'},
                 title='Neighbourhoods Crime Count Yearly')
       st.plotly_chart(fig)

    # Neighbourhoods Crime Count Yearly Trend Line Plot
       neighbourhood_crime_type_yearly_trend = df.groupby(['NEIGHBOURHOOD', 'YEAR']).size().reset_index().rename(columns={0: 'Crime_Count'})
       st.subheader("Neighbourhoods Crime Count Yearly Trend")
       fig = px.line(neighbourhood_crime_type_yearly_trend, x='YEAR', y='Crime_Count', color='NEIGHBOURHOOD',
                 hover_data= {'YEAR': True, 'Crime_Count': True},
                 labels={'Crime_Count': 'Number of Crimes', 'YEAR': 'Year'},
                 title='Neighbourhoods Crime Count Yearly Trend')
       st.plotly_chart(fig)

    # Neighbourhoods Crime Type Counts Yearly Heatmap
       neighbourhood_crime_type_count_yearly = df.groupby(['NEIGHBOURHOOD', 'TYPE', 'YEAR']).size().reset_index().rename(columns={0: 'Crime_Count'})
       st.subheader("Neighbourhoods Crime Type Counts Yearly")
       fig = px.density_heatmap(neighbourhood_crime_type_count_yearly, x='NEIGHBOURHOOD', y='TYPE', z='Crime_Count', facet_col='YEAR', facet_col_wrap=3,
                 hover_data= {'TYPE': True, 'YEAR': True, 'Crime_Count': True},
                 labels={'Crime_Count': 'Number of Crimes', 'NEIGHBOURHOOD': 'Neighbourhood'},
                 title='Neighbourhoods Crime Type Counts Yearly')
       fig.update_layout(title_x=0.5, title_font_family="Arial", height=1200, width=1400)
       st.plotly_chart(fig)

    # Neighbourhoods Crime Type Counts Yearly Line Plot
       neighbourhood_crime_type_yearly_trend = df.groupby(['NEIGHBOURHOOD', 'TYPE', 'YEAR']).size().reset_index().rename(columns={0: 'Crime_Count'})
       st.subheader("Neighbourhoods Crime Type Counts Yearly Trend")
       fig = px.line(neighbourhood_crime_type_yearly_trend, x='YEAR', y='Crime_Count', color='NEIGHBOURHOOD', facet_col='TYPE', facet_col_wrap=3,
                 hover_data= {'TYPE': True, 'YEAR': True, 'Crime_Count': True},
                 labels={'Crime_Count': 'Number of Crimes', 'YEAR': 'Year'},
                 title='Neighbourhoods Crime Types Yearly Trend')
       fig.update_layout(height=800, width=1200)
       st.plotly_chart(fig)

    # Neighbourhoods Crime Count Monthly Bar Plot
       neighbourhood_crime_count_monthly = df.groupby(['NEIGHBOURHOOD', 'YEAR', 'MONTH']).size().reset_index().rename(columns={0: 'Crime_Count'})
       st.subheader("Neighbourhoods Crime Count Monthly")
       fig = px.bar(neighbourhood_crime_count_monthly, x='NEIGHBOURHOOD', y='Crime_Count', color='YEAR', facet_col='MONTH', facet_col_wrap=4,
                 hover_data= {'MONTH': True, 'YEAR': True, 'Crime_Count': True},
                 labels={'Crime_Count': 'Number of Crimes', 'NEIGHBOURHOOD': 'Neighbourhood'},
                 title='Neighbourhoods Crime Count Monthly')
       fig.update_layout(height=800, width=1200)
       st.plotly_chart(fig)

    # Neighbourhoods Crime Count Monthly Trend Line Plot
       neighbourhood_crime_Count_monthly_trend = df.groupby(['NEIGHBOURHOOD', 'YEAR', 'MONTH']).size().reset_index().rename(columns={0: 'Crime_Count'})
       st.subheader("Neighbourhoods Crime Count Monthly Trend")
       fig = px.line(neighbourhood_crime_Count_monthly_trend, x='MONTH', y='Crime_Count', color='NEIGHBOURHOOD', facet_col='YEAR', facet_col_wrap=4,
                 hover_data= {'MONTH': True, 'YEAR': True, 'Crime_Count': True},
                 labels={'Crime_Count': 'Number of Crimes', 'MONTH': 'Month'},
                 title='Neighbourhoods Crime Count Monthly Trend')
       fig.update_layout(height=800, width=1200)
       st.plotly_chart(fig)

    # Neighbourhoods Crime Type Counts Monthly Bar Plot
       neighbourhood_crime_type_count_monthly = df.groupby(['NEIGHBOURHOOD', 'TYPE', 'YEAR', 'MONTH']).size().reset_index().rename(columns={0: 'Crime_Count'})
       st.subheader("Neighbourhoods Crime Type Counts Monthly")
       fig = px.bar(neighbourhood_crime_type_count_monthly, x='NEIGHBOURHOOD', y='Crime_Count', color='TYPE', facet_col='MONTH', facet_col_wrap=3,
                 hover_data= {'TYPE': True, 'MONTH': True, 'YEAR': True, 'Crime_Count': True},
                 labels={'Crime_Count': 'Number of Crimes', 'NEIGHBOURHOOD': 'Neighbourhood'},
                 title='Neighbourhoods Crime Type Counts Monthly')
       fig.update_layout(height=800, width=1200)
       st.plotly_chart(fig)

    # Neighbourhoods Crime Type Counts Monthly Line Plot
       neighbourhood_crime_type_monthly_trend = df.groupby(['NEIGHBOURHOOD', 'TYPE', 'YEAR', 'MONTH']).size().reset_index().rename(columns={0: 'Crime_Count'})
       st.subheader("Neighbourhoods Crime Type Counts Monthly Trend")
       fig = px.line(neighbourhood_crime_type_monthly_trend, x='YEAR', y='Crime_Count', color='NEIGHBOURHOOD', facet_col='MONTH', facet_col_wrap=3,
                 hover_data= {'TYPE': True, 'MONTH': True, 'YEAR': True, 'Crime_Count': True},
                 labels={'Crime_Count': 'Number of Crimes', 'MONTH': 'Month'},
                 title='Neighbourhoods Crime Type Counts Monthly Trend')
       fig.update_layout(height=800, width=1200)
       st.plotly_chart(fig)

    elif visual_selection == "Yearly Crime Trend":
       st.write("Yearly Crime Trend Line Plot is displayed below:")
       yearly_crime_count = df.groupby('YEAR')['TYPE'].count()
       st.subheader("Yearly Crime Trend")
       fig = px.line(x=yearly_crime_count.index, y=yearly_crime_count.values,
                  labels={'x': 'Year', 'y': 'Number of Crimes'},
                  title='Yearly Crime Trend')
       st.plotly_chart(fig)

    elif visual_selection == "Yearly Crime Trend by Type":
       st.write("Yearly Crime Count Bar Plot is displayed below:")
       yearly_crime_count_per_type = df.groupby(['YEAR', 'TYPE']).size().reset_index().rename(columns={0: 'Count'})
       st.subheader("Yearly Crime Count by Type")
       fig = px.bar(yearly_crime_count_per_type, x='YEAR',
                 y='Count', color='TYPE', hover_data= ['TYPE', 'YEAR', 'Count'],
                 labels={'Count': 'Number of Crimes', 'YEAR': 'Year'},
                 title='Yearly Crime Trend by Type')
       st.plotly_chart(fig)

    elif visual_selection == "Monthly Crime Trend":

       st.write("Monthly Crime Trend Line Plot is displayed below:")
       monthly_crime_count = df.groupby(['YEAR', 'MONTH']).size().reset_index().rename(columns={0: 'Count'})
       monthly_crime_count_df = monthly_crime_count.copy()
       monthly_crime_count_df['Month_name'] = monthly_crime_count_df['MONTH'].apply(lambda x: pd.to_datetime(str(x), format='%m').strftime('%B'))
       st.subheader("Monthly Crime Trend")
       fig = px.line(monthly_crime_count_df, x='Month_name', y='Count', color='YEAR',
                  labels={'Month_name': 'Month', 'Count': 'Number of Crimes', 'YEAR': 'Year'},
                  title='Monthly Crime Trend')
       st.plotly_chart(fig)

    elif visual_selection == "Monthly Crime Trend by Type":
       st.write("Monthly Crime Count Bar Plot is displayed below:")
       monthly_crime_count_per_type = df.groupby(['TYPE', 'YEAR', 'MONTH']).size().reset_index().rename(columns={0: 'Count'})
       monthly_crime_count_per_type_df = monthly_crime_count_per_type.copy()
       monthly_crime_count_per_type_df['Month_name'] = monthly_crime_count_per_type_df['MONTH'].apply(lambda x: pd.to_datetime(str(x), format='%m').strftime('%B'))
       st.subheader("Monthly Crime Count by Type")
       fig = px.bar(monthly_crime_count_per_type_df, x='Month_name', y='Count', color='TYPE', facet_col='YEAR', facet_col_wrap=4,
                 hover_data= ['TYPE', 'MONTH', 'Count'],
                 labels={'Count': 'Number of Crimes', 'Month_name': 'Month'},
                 title='Monthly Crime Trend by Type')
       fig.update_layout(height=800, width=1200)
       st.plotly_chart(fig)     

    elif visual_selection == "Crime Location Map":

       neighbourhood_crime_count = df.groupby(['NEIGHBOURHOOD'])['TYPE'].count().reset_index().rename(columns={'TYPE': 'Crime_Count'})
       latitude_longitude_df = df.groupby(['NEIGHBOURHOOD'])[['Latitude', 'Longitude']].mean().reset_index()
       crime_location_df = pd.merge(neighbourhood_crime_count, latitude_longitude_df, on=['NEIGHBOURHOOD'], how='inner')
       crime_map = folium.Map(location=[crime_location_df['Latitude'].mean(), crime_location_df['Longitude'].mean()], zoom_start=12)
       for _, row in crime_location_df.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=row['Crime_Count'] * 0.001,
            popup=f"{row['NEIGHBOURHOOD']}: {row['Crime_Count']} crimes",
            color='blue',
            fill=True,
            fill_color='blue'
        ).add_to(crime_map)
       st.subheader("Crime Location Map")
       st_folium = st.components.v1.html(crime_map._repr_html_(), width=1000, height=500)

       selected_neighbourhood = st.sidebar.selectbox(
         "Select Neighbourhood",
         df["NEIGHBOURHOOD"].unique())
       
       if selected_neighbourhood == "Arbutus Ridge":
          st.write("You have selected Arbutus Ridge. Displaying crime details for this neighbourhood.")
          arbutus_ridge_crimes = df[df["NEIGHBOURHOOD"] == "Arbutus Ridge"]
          arbutus_ridge_lat_long = arbutus_ridge_crimes.groupby('HUNDRED_BLOCK')[['Latitude', 'Longitude']].mean().reset_index()
          crime_location_df = pd.merge(arbutus_ridge_crimes, arbutus_ridge_lat_long, on='HUNDRED_BLOCK', how='inner')
          crime_map = folium.Map(location=[crime_location_df['Latitude'].mean(), crime_location_df['Longitude'].mean()], zoom_start=13)
          for _, row in crime_location_df.itertuples():
                 folium.CircleMarker(
                     location=[row['Latitude'], row['Longitude']],
                     radius=5,
                     popup=f"{row['HUNDRED_BLOCK']}: {row['TYPE']} crime",
                     color='red',
                     fill=True,
                     fill_color='red'
                 ).add_to(crime_map)
          st.subheader("Crime Location Map for Arbutus Ridge")
          st.components.v1.html(crime_map._repr_html_(), width=1000, height=500)
          st.dataframe(arbutus_ridge_crimes)

       elif selected_neighbourhood == "Central Business District":
          st.write("You have selected Central Business District. Displaying crime details for this neighbourhood.")
          cbd_crimes = df[df["NEIGHBOURHOOD"] == "Central Business District"]
          cbd_lat_long = cbd_crimes.groupby('HUNDRED_BLOCK')[['Latitude', 'Longitude']].mean().reset_index()
          crime_location_df = pd.merge(cbd_crimes, cbd_lat_long, on='HUNDRED_BLOCK', how='inner')
          crime_map = folium.Map(location=[crime_location_df['Latitude'].mean(), crime_location_df['Longitude'].mean()], zoom_start=13)
          for _, row in crime_location_df.itertuples():
                 folium.CircleMarker(
                     location=[row['Latitude'], row['Longitude']],
                     radius=5,
                     popup=f"{row['HUNDRED_BLOCK']}: {row['TYPE']} crime",
                     color='red',
                     fill=True,
                     fill_color='red'
                 ).add_to(crime_map)
          st.subheader("Crime Location Map for Central Business District")
          st.components.v1.html(crime_map._repr_html_(), width=1000, height=500)
          st.dataframe(cbd_crimes)

       elif selected_neighbourhood == "Dunbar-Southlands":
          st.write("You have selected Dunbar-Southlands. Displaying crime details for this neighbourhood.")
          dunbar_southlands_crimes = df[df["NEIGHBOURHOOD"] == "Dunbar-Southlands"]
          dunbar_southlands_lat_long = dunbar_southlands_crimes.groupby('HUNDRED_BLOCK')[['Latitude', 'Longitude']].mean().reset_index()
          crime_location_df = pd.merge(dunbar_southlands_crimes, dunbar_southlands_lat_long, on='HUNDRED_BLOCK', how='inner')
          crime_map = folium.Map(location=[crime_location_df['Latitude'].mean(), crime_location_df['Longitude'].mean()], zoom_start=13)
          for _, row in crime_location_df.itertuples():
                 folium.CircleMarker(
                     location=[row['Latitude'], row['Longitude']],
                     radius=5,
                     popup=f"{row['HUNDRED_BLOCK']}: {row['TYPE']} crime",
                     color='red',
                     fill=True,
                     fill_color='red'
                 ).add_to(crime_map)
          st.subheader("Crime Location Map for Dunbar-Southlands")
          st.components.v1.html(crime_map._repr_html_(), width=1000, height=500)
          st.dataframe(dunbar_southlands_crimes)

       elif selected_neighbourhood == "Fairview":
          st.write("You have selected Fairview. Displaying crime details for this neighbourhood.")
          fairview_crimes = df[df["NEIGHBOURHOOD"] == "Fairview"]
          fairview_lat_long = fairview_crimes.groupby('HUNDRED_BLOCK')[['Latitude', 'Longitude']].mean().reset_index()
          crime_location_df = pd.merge(fairview_crimes, fairview_lat_long, on='HUNDRED_BLOCK', how='inner')
          crime_map = folium.Map(location=[crime_location_df['Latitude'].mean(), crime_location_df['Longitude'].mean()], zoom_start=13)
          for _, row in crime_location_df.itertuples():
                 folium.CircleMarker(
                     location=[row['Latitude'], row['Longitude']],
                     radius=5,
                     popup=f"{row['HUNDRED_BLOCK']}: {row['TYPE']} crime",
                     color='red',
                     fill=True,
                     fill_color='red'
                 ).add_to(crime_map)
          st.subheader("Crime Location Map for Fairview")
          st.components.v1.html(crime_map._repr_html_(), width=1000, height=500)
          st.dataframe(fairview_crimes)

       elif selected_neighbourhood == "Grandview-Woodland":
          st.write("You have selected Grandview-Woodland. Displaying crime details for this neighbourhood.")
          grandview_woodland_crimes = df[df["NEIGHBOURHOOD"] == "Grandview-Woodland"]
          grandview_woodland_lat_long = grandview_woodland_crimes.groupby('HUNDRED_BLOCK')[['Latitude', 'Longitude']].mean().reset_index()
          crime_location_df = pd.merge(grandview_woodland_crimes, grandview_woodland_lat_long, on='HUNDRED_BLOCK', how='inner')
          crime_map = folium.Map(location=[crime_location_df['Latitude'].mean(), crime_location_df['Longitude'].mean()], zoom_start=13)
          for _, row in crime_location_df.itertuples():
                 folium.CircleMarker(
                     location=[row['Latitude'], row['Longitude']],
                     radius=5,
                     popup=f"{row['HUNDRED_BLOCK']}: {row['TYPE']} crime",
                     color='red',
                     fill=True,
                     fill_color='red'
                 ).add_to(crime_map)
          st.subheader("Crime Location Map for Grandview-Woodland")
          st.components.v1.html(crime_map._repr_html_(), width=1000, height=500)
          st.dataframe(grandview_woodland_crimes)

       elif selected_neighbourhood == "Hastings-Sunrise":
          st.write("You have selected Hastings-Sunrise. Displaying crime details for this neighbourhood.")
          hastings_sunrise_crimes = df[df["NEIGHBOURHOOD"] == "Hastings-Sunrise"]
          hastings_sunrise_lat_long = hastings_sunrise_crimes.groupby('HUNDRED_BLOCK')[['Latitude', 'Longitude']].mean().reset_index()
          crime_location_df = pd.merge(hastings_sunrise_crimes, hastings_sunrise_lat_long, on='HUNDRED_BLOCK', how='inner')
          crime_map = folium.Map(location=[crime_location_df['Latitude'].mean(), crime_location_df['Longitude'].mean()], zoom_start=13)
          for _, row in crime_location_df.itertuples():
                 folium.CircleMarker(
                     location=[row['Latitude'], row['Longitude']],
                     radius=5,
                     popup=f"{row['HUNDRED_BLOCK']}: {row['TYPE']} crime",
                     color='red',
                     fill=True,
                     fill_color='red'
                 ).add_to(crime_map)
          st.subheader("Crime Location Map for Hastings-Sunrise")
          st.components.v1.html(crime_map._repr_html_(), width=1000, height=500)
          st.dataframe(hastings_sunrise_crimes)

       elif selected_neighbourhood == "Kensington-Cedar Cottage":
          st.write("You have selected Kensington-Cedar Cottage. Displaying crime details for this neighbourhood.")
          kensington_cedar_cottage_crimes = df[df["NEIGHBOURHOOD"] == "Kensington-Cedar Cottage"]
          kensington_cedar_cottage_lat_long = kensington_cedar_cottage_crimes.groupby('HUNDRED_BLOCK')[['Latitude', 'Longitude']].mean().reset_index()
          crime_location_df = pd.merge(kensington_cedar_cottage_crimes, kensington_cedar_cottage_lat_long, on='HUNDRED_BLOCK', how='inner')
          crime_map = folium.Map(location=[crime_location_df['Latitude'].mean(), crime_location_df['Longitude'].mean()], zoom_start=13)
          for _, row in crime_location_df.itertuples():
                 folium.CircleMarker(
                     location=[row['Latitude'], row['Longitude']],
                     radius=5,
                     popup=f"{row['HUNDRED_BLOCK']}: {row['TYPE']} crime",
                     color='red',
                     fill=True,
                     fill_color='red'
                 ).add_to(crime_map)
          st.subheader("Crime Location Map for Kensington-Cedar Cottage")
          st.components.v1.html(crime_map._repr_html_(), width=1000, height=500)
          st.dataframe(kensington_cedar_cottage_crimes)

       elif selected_neighbourhood == "Kerrisdale":
          st.write("You have selected Kerrisdale. Displaying crime details for this neighbourhood.")
          kerrisdale_crimes = df[df["NEIGHBOURHOOD"] == "Kerrisdale"]
          kerrisdale_lat_long = kerrisdale_crimes.groupby('HUNDRED_BLOCK')[['Latitude', 'Longitude']].mean().reset_index()
          crime_location_df = pd.merge(kerrisdale_crimes, kerrisdale_lat_long, on='HUNDRED_BLOCK', how='inner')
          crime_map = folium.Map(location=[crime_location_df['Latitude'].mean(), crime_location_df['Longitude'].mean()], zoom_start=13)
          for _, row in crime_location_df.itertuples():
                 folium.CircleMarker(
                     location=[row['Latitude'], row['Longitude']],
                     radius=5,
                     popup=f"{row['HUNDRED_BLOCK']}: {row['TYPE']} crime",
                     color='red',
                     fill=True,
                     fill_color='red'
                 ).add_to(crime_map)
          st.subheader("Crime Location Map for Kerrisdale")
          st.components.v1.html(crime_map._repr_html_(), width=1000, height=500)
          st.dataframe(kerrisdale_crimes)

       elif selected_neighbourhood == "Killarney":
          st.write("You have selected Killarney. Displaying crime details for this neighbourhood.")
          killarney_crimes = df[df["NEIGHBOURHOOD"] == "Killarney"]
          killarney_lat_long = killarney_crimes.groupby('HUNDRED_BLOCK')[['Latitude', 'Longitude']].mean().reset_index()
          crime_location_df = pd.merge(killarney_crimes, killarney_lat_long, on='HUNDRED_BLOCK', how='inner')
          crime_map = folium.Map(location=[crime_location_df['Latitude'].mean(), crime_location_df['Longitude'].mean()], zoom_start=13)
          for _, row in crime_location_df.itertuples():
                 folium.CircleMarker(
                     location=[row['Latitude'], row['Longitude']],
                     radius=5,
                     popup=f"{row['HUNDRED_BLOCK']}: {row['TYPE']} crime",
                     color='red',
                     fill=True,
                     fill_color='red'
                 ).add_to(crime_map)
          st.subheader("Crime Location Map for Killarney")
          st.components.v1.html(crime_map._repr_html_(), width=1000, height=500)
          st.dataframe(killarney_crimes)

       elif selected_neighbourhood == "Kitsilano":
          st.write("You have selected Kitsilano. Displaying crime details for this neighbourhood.")
          kitsilano_crimes = df[df["NEIGHBOURHOOD"] == "Kitsilano"]
          kitsilano_lat_long = kitsilano_crimes.groupby('HUNDRED_BLOCK')[['Latitude', 'Longitude']].mean().reset_index()
          crime_location_df = pd.merge(kitsilano_crimes, kitsilano_lat_long, on='HUNDRED_BLOCK', how='inner')
          crime_map = folium.Map(location=[crime_location_df['Latitude'].mean(), crime_location_df['Longitude'].mean()], zoom_start=13)
          for _, row in crime_location_df.itertuples():
                 folium.CircleMarker(
                     location=[row['Latitude'], row['Longitude']],
                     radius=5,
                     popup=f"{row['HUNDRED_BLOCK']}: {row['TYPE']} crime",
                     color='red',
                     fill=True,
                     fill_color='red'
                 ).add_to(crime_map)
          st.subheader("Crime Location Map for Kitsilano")
          st.components.v1.html(crime_map._repr_html_(), width=1000, height=500)
          st.dataframe(kitsilano_crimes)

       elif selected_neighbourhood == "Marpole":
          st.write("You have selected Marpole. Displaying crime details for this neighbourhood.")
          marpole_crimes = df[df["NEIGHBOURHOOD"] == "Marpole"]
          marpole_lat_long = marpole_crimes.groupby('HUNDRED_BLOCK')[['Latitude', 'Longitude']].mean().reset_index()
          crime_location_df = pd.merge(marpole_crimes, marpole_lat_long, on='HUNDRED_BLOCK', how='inner')
          crime_map = folium.Map(location=[crime_location_df['Latitude'].mean(), crime_location_df['Longitude'].mean()], zoom_start=13)
          for _, row in crime_location_df.itertuples():
                 folium.CircleMarker(
                     location=[row['Latitude'], row['Longitude']],
                     radius=5,
                     popup=f"{row['HUNDRED_BLOCK']}: {row['TYPE']} crime",
                     color='red',
                     fill=True,
                     fill_color='red'
                 ).add_to(crime_map)
          st.subheader("Crime Location Map for Marpole")
          st.components.v1.html(crime_map._repr_html_(), width=1000, height=500)
          st.dataframe(marpole_crimes)

       elif selected_neighbourhood == "Mount Pleasant":
          st.write("You have selected Mount Pleasant. Displaying crime details for this neighbourhood.")
          mount_pleasant_crimes = df[df["NEIGHBOURHOOD"] == "Mount Pleasant"]
          mount_pleasant_lat_long = mount_pleasant_crimes.groupby('HUNDRED_BLOCK')[['Latitude', 'Longitude']].mean().reset_index()
          crime_location_df = pd.merge(mount_pleasant_crimes, mount_pleasant_lat_long, on='HUNDRED_BLOCK', how='inner')
          crime_map = folium.Map(location=[crime_location_df['Latitude'].mean(), crime_location_df['Longitude'].mean()], zoom_start=13)
          for _, row in crime_location_df.itertuples():
                 folium.CircleMarker(
                     location=[row['Latitude'], row['Longitude']],
                     radius=5,
                     popup=f"{row['HUNDRED_BLOCK']}: {row['TYPE']} crime",
                     color='red',
                     fill=True,
                     fill_color='red'
                 ).add_to(crime_map)
          st.subheader("Crime Location Map for Mount Pleasant")
          st.components.v1.html(crime_map._repr_html_(), width=1000, height=500)
          st.dataframe(mount_pleasant_crimes)

       elif selected_neighbourhood == "Musqueam":
          st.write("You have selected Musqueam. Displaying crime details for this neighbourhood.")
          musqueam_crimes = df[df["NEIGHBOURHOOD"] == "Musqueam"]
          musqueam_lat_long = musqueam_crimes.groupby('HUNDRED_BLOCK')[['Latitude', 'Longitude']].mean().reset_index()
          crime_location_df = pd.merge(musqueam_crimes, musqueam_lat_long, on='HUNDRED_BLOCK', how='inner')
          crime_map = folium.Map(location=[crime_location_df['Latitude'].mean(), crime_location_df['Longitude'].mean()], zoom_start=13)
          for _, row in crime_location_df.itertuples():
                 folium.CircleMarker(
                     location=[row['Latitude'], row['Longitude']],
                     radius=5,
                     popup=f"{row['HUNDRED_BLOCK']}: {row['TYPE']} crime",
                     color='red',
                     fill=True,
                     fill_color='red'
                 ).add_to(crime_map)
          st.subheader("Crime Location Map for Musqueam")
          st.components.v1.html(crime_map._repr_html_(), width=1000, height=500)
          st.dataframe(musqueam_crimes)

       elif selected_neighbourhood == "Oakridge":
          st.write("You have selected Oakridge. Displaying crime details for this neighbourhood.")
          oakridge_crimes = df[df["NEIGHBOURHOOD"] == "Oakridge"]
          oakridge_lat_long = oakridge_crimes.groupby('HUNDRED_BLOCK')[['Latitude', 'Longitude']].mean().reset_index()
          crime_location_df = pd.merge(oakridge_crimes, oakridge_lat_long, on='HUNDRED_BLOCK', how='inner')
          crime_map = folium.Map(location=[crime_location_df['Latitude'].mean(), crime_location_df['Longitude'].mean()], zoom_start=13)
          for _, row in crime_location_df.itertuples():
                 folium.CircleMarker(
                     location=[row['Latitude'], row['Longitude']],
                     radius=5,
                     popup=f"{row['HUNDRED_BLOCK']}: {row['TYPE']} crime",
                     color='red',
                     fill=True,
                     fill_color='red'
                 ).add_to(crime_map)
          st.subheader("Crime Location Map for Oakridge")
          st.components.v1.html(crime_map._repr_html_(), width=1000, height=500)
          st.dataframe(oakridge_crimes)

       elif selected_neighbourhood == "Renfrew-Collingwood":
          st.write("You have selected Renfrew-Collingwood. Displaying crime details for this neighbourhood.")
          renfrew_collingwood_crimes = df[df["NEIGHBOURHOOD"] == "Renfrew-Collingwood"]
          renfrew_collingwood_lat_long = renfrew_collingwood_crimes.groupby('HUNDRED_BLOCK')[['Latitude', 'Longitude']].mean().reset_index()
          crime_location_df = pd.merge(renfrew_collingwood_crimes, renfrew_collingwood_lat_long, on='HUNDRED_BLOCK', how='inner')
          crime_map = folium.Map(location=[crime_location_df['Latitude'].mean(), crime_location_df['Longitude'].mean()], zoom_start=13)
          for _, row in crime_location_df.itertuples():
                 folium.CircleMarker(
                     location=[row['Latitude'], row['Longitude']],
                     radius=5,
                     popup=f"{row['HUNDRED_BLOCK']}: {row['TYPE']} crime",
                     color='red',
                     fill=True,
                     fill_color='red'
                 ).add_to(crime_map)
          st.subheader("Crime Location Map for Renfrew-Collingwood")
          st.components.v1.html(crime_map._repr_html_(), width=1000, height=500)
          st.dataframe(renfrew_collingwood_crimes)

       elif selected_neighbourhood == "Riley Park":
          st.write("You have selected Riley Park. Displaying crime details for this neighbourhood.")
          riley_park_crimes = df[df["NEIGHBOURHOOD"] == "Riley Park"]
          riley_park_lat_long = riley_park_crimes.groupby('HUNDRED_BLOCK')[['Latitude', 'Longitude']].mean().reset_index()
          crime_location_df = pd.merge(riley_park_crimes, riley_park_lat_long, on='HUNDRED_BLOCK', how='inner')
          crime_map = folium.Map(location=[crime_location_df['Latitude'].mean(), crime_location_df['Longitude'].mean()], zoom_start=13)
          for _, row in crime_location_df.itertuples():
                 folium.CircleMarker(
                     location=[row['Latitude'], row['Longitude']],
                     radius=5,
                     popup=f"{row['HUNDRED_BLOCK']}: {row['TYPE']} crime",
                     color='red',
                     fill=True,
                     fill_color='red'
                 ).add_to(crime_map)
          st.subheader("Crime Location Map for Riley Park")
          st.components.v1.html(crime_map._repr_html_(), width=1000, height=500)
          st.dataframe(riley_park_crimes)

       elif selected_neighbourhood == "Shaughnessy":
          st.write("You have selected Shaughnessy. Displaying crime details for this neighbourhood.")
          shaughnessy_crimes = df[df["NEIGHBOURHOOD"] == "Shaughnessy"]
          shaughnessy_lat_long = shaughnessy_crimes.groupby('HUNDRED_BLOCK')[['Latitude', 'Longitude']].mean().reset_index()
          crime_location_df = pd.merge(shaughnessy_crimes, shaughnessy_lat_long, on='HUNDRED_BLOCK', how='inner')
          crime_map = folium.Map(location=[crime_location_df['Latitude'].mean(), crime_location_df['Longitude'].mean()], zoom_start=13)
          for _, row in crime_location_df.itertuples():
                 folium.CircleMarker(
                     location=[row['Latitude'], row['Longitude']],
                     radius=5,
                     popup=f"{row['HUNDRED_BLOCK']}: {row['TYPE']} crime",
                     color='red',
                     fill=True,
                     fill_color='red'
                 ).add_to(crime_map)
          st.subheader("Crime Location Map for Shaughnessy")
          st.components.v1.html(crime_map._repr_html_(), width=1000, height=500)
          st.dataframe(shaughnessy_crimes)

       elif selected_neighbourhood == "South Cambie":
          st.write("You have selected South Cambie. Displaying crime details for this neighbourhood.")
          south_cambie_crimes = df[df["NEIGHBOURHOOD"] == "South Cambie"]
          south_cambie_lat_long = south_cambie_crimes.groupby('HUNDRED_BLOCK')[['Latitude', 'Longitude']].mean().reset_index()
          crime_location_df = pd.merge(south_cambie_crimes, south_cambie_lat_long, on='HUNDRED_BLOCK', how='inner')
          crime_map = folium.Map(location=[crime_location_df['Latitude'].mean(), crime_location_df['Longitude'].mean()], zoom_start=13)
          for _, row in crime_location_df.itertuples():
                 folium.CircleMarker(
                     location=[row['Latitude'], row['Longitude']],
                     radius=5,
                     popup=f"{row['HUNDRED_BLOCK']}: {row['TYPE']} crime",
                     color='red',
                     fill=True,
                     fill_color='red'
                 ).add_to(crime_map)
          st.subheader("Crime Location Map for South Cambie")
          st.components.v1.html(crime_map._repr_html_(), width=1000, height=500)
          st.dataframe(south_cambie_crimes)

       elif selected_neighbourhood == "Stanley Park":
          st.write("You have selected Stanley Park. Displaying crime details for this neighbourhood.")
          stanley_park_crimes = df[df["NEIGHBOURHOOD"] == "Stanley Park"]
          stanley_park_lat_long = stanley_park_crimes.groupby('HUNDRED_BLOCK')[['Latitude', 'Longitude']].mean().reset_index()
          crime_location_df = pd.merge(stanley_park_crimes, stanley_park_lat_long, on='HUNDRED_BLOCK', how='inner')
          crime_map = folium.Map(location=[crime_location_df['Latitude'].mean(), crime_location_df['Longitude'].mean()], zoom_start=13)
          for _, row in crime_location_df.itertuples():
                 folium.CircleMarker(
                     location=[row['Latitude'], row['Longitude']],
                     radius=5,
                     popup=f"{row['HUNDRED_BLOCK']}: {row['TYPE']} crime",
                     color='red',
                     fill=True,
                     fill_color='red'
                 ).add_to(crime_map)
          st.subheader("Crime Location Map for Stanley Park")
          st.components.v1.html(crime_map._repr_html_(), width=1000, height=500)
          st.dataframe(stanley_park_crimes)

       elif selected_neighbourhood == "Strathcona":
          st.write("You have selected Strathcona. Displaying crime details for this neighbourhood.")
          strathcona_crimes = df[df["NEIGHBOURHOOD"] == "Strathcona"]
          strathcona_lat_long = strathcona_crimes.groupby('HUNDRED_BLOCK')[['Latitude', 'Longitude']].mean().reset_index()
          crime_location_df = pd.merge(strathcona_crimes, strathcona_lat_long, on='HUNDRED_BLOCK', how='inner')
          crime_map = folium.Map(location=[crime_location_df['Latitude'].mean(), crime_location_df['Longitude'].mean()], zoom_start=13)
          for _, row in crime_location_df.itertuples():
                 folium.CircleMarker(
                     location=[row['Latitude'], row['Longitude']],
                     radius=5,
                     popup=f"{row['HUNDRED_BLOCK']}: {row['TYPE']} crime",
                     color='red',
                     fill=True,
                     fill_color='red'
                 ).add_to(crime_map)
          st.subheader("Crime Location Map for Strathcona")
          st.components.v1.html(crime_map._repr_html_(), width=1000, height=500)
          st.dataframe(strathcona_crimes)

       elif selected_neighbourhood == "Sunset":
          st.write("You have selected Sunset. Displaying crime details for this neighbourhood.")
          sunset_crimes = df[df["NEIGHBOURHOOD"] == "Sunset"]
          sunset_lat_long = sunset_crimes.groupby('HUNDRED_BLOCK')[['Latitude', 'Longitude']].mean().reset_index()
          crime_location_df = pd.merge(sunset_crimes, sunset_lat_long, on='HUNDRED_BLOCK', how='inner')
          crime_map = folium.Map(location=[crime_location_df['Latitude'].mean(), crime_location_df['Longitude'].mean()], zoom_start=13)
          for _, row in crime_location_df.itertuples():
                 folium.CircleMarker(
                     location=[row['Latitude'], row['Longitude']],
                     radius=5,
                     popup=f"{row['HUNDRED_BLOCK']}: {row['TYPE']} crime",
                     color='red',
                     fill=True,
                     fill_color='red'
                 ).add_to(crime_map)
          st.subheader("Crime Location Map for Sunset")
          st.components.v1.html(crime_map._repr_html_(), width=1000, height=500)
          st.dataframe(sunset_crimes)

       elif selected_neighbourhood == "Victoria-Fraserview":
          st.write("You have selected Victoria-Fraserview. Displaying crime details for this neighbourhood.")
          victoria_fraserview_crimes = df[df["NEIGHBOURHOOD"] == "Victoria-Fraserview"]
          victoria_fraserview_lat_long = victoria_fraserview_crimes.groupby('HUNDRED_BLOCK')[['Latitude', 'Longitude']].mean().reset_index()
          crime_location_df = pd.merge(victoria_fraserview_crimes, victoria_fraserview_lat_long, on='HUNDRED_BLOCK', how='inner')
          crime_map = folium.Map(location=[crime_location_df['Latitude'].mean(), crime_location_df['Longitude'].mean()], zoom_start=13)
          for _, row in crime_location_df.itertuples():
                 folium.CircleMarker(
                     location=[row['Latitude'], row['Longitude']],
                     radius=5,
                     popup=f"{row['HUNDRED_BLOCK']}: {row['TYPE']} crime",
                     color='red',
                     fill=True,
                     fill_color='red'
                 ).add_to(crime_map)
          st.subheader("Crime Location Map for Victoria-Fraserview")
          st.components.v1.html(crime_map._repr_html_(), width=1000, height=500)
          st.dataframe(victoria_fraserview_crimes)

       elif selected_neighbourhood == "West End":
          st.write("You have selected West End. Displaying crime details for this neighbourhood.")
          west_end_crimes = df[df["NEIGHBOURHOOD"] == "West End"]
          west_end_lat_long = west_end_crimes.groupby('HUNDRED_BLOCK')[['Latitude', 'Longitude']].mean().reset_index()
          crime_location_df = pd.merge(west_end_crimes, west_end_lat_long, on='HUNDRED_BLOCK', how='inner')
          crime_map = folium.Map(location=[crime_location_df['Latitude'].mean(), crime_location_df['Longitude'].mean()], zoom_start=13)
          for _, row in crime_location_df.itertuples():
                 folium.CircleMarker(
                     location=[row['Latitude'], row['Longitude']],
                     radius=5,
                     popup=f"{row['HUNDRED_BLOCK']}: {row['TYPE']} crime",
                     color='red',
                     fill=True,
                     fill_color='red'
                 ).add_to(crime_map)
          st.subheader("Crime Location Map for West End")
          st.components.v1.html(crime_map._repr_html_(), width=1000, height=500)
          st.dataframe(west_end_crimes)

       elif selected_neighbourhood == "West Point Grey":
          st.write("You have selected West Point Grey. Displaying crime details for this neighbourhood.")
          west_point_grey_crimes = df[df["NEIGHBOURHOOD"] == "West Point Grey"]
          west_point_grey_lat_long = west_point_grey_crimes.groupby('HUNDRED_BLOCK')[['Latitude', 'Longitude']].mean().reset_index()
          crime_location_df = pd.merge(west_point_grey_crimes, west_point_grey_lat_long, on='HUNDRED_BLOCK', how='inner')
          crime_map = folium.Map(location=[crime_location_df['Latitude'].mean(), crime_location_df['Longitude'].mean()], zoom_start=13)
          for _, row in crime_location_df.itertuples():
                 folium.CircleMarker(
                     location=[row['Latitude'], row['Longitude']],
                     radius=5,
                     popup=f"{row['HUNDRED_BLOCK']}: {row['TYPE']} crime",
                     color='red',
                     fill=True,
                     fill_color='red'
                 ).add_to(crime_map)
          st.subheader("Crime Location Map for West Point Grey")
          st.components.v1.html(crime_map._repr_html_(), width=1000, height=500)
          st.dataframe(west_point_grey_crimes)            

    elif visual_selection == "Neighbourhoods Vs Crime Count":
       st.write("Neighbourhoods Vs Crime Count Bar Plot is displayed below:")
       neighbourhood_crime_totals = df['NEIGHBOURHOOD'].value_counts().reset_index()
       neighbourhood_crime_totals.columns = ['NEIGHBOURHOOD', 'Crime_Count']
       st.subheader("Neighbourhoods Vs Crime Count")
       fig = px.bar(neighbourhood_crime_totals, x='NEIGHBOURHOOD', y='Crime_Count',
                 labels={'NEIGHBOURHOOD': 'Neighbourhood', 'Crime_Count': 'Number of Crimes'},
                 title='Neighbourhoods Vs Crime Count')
       st.plotly_chart(fig)

    elif visual_selection == "Neighbourhoods Vs Crime Type Heatmap":
       st.write("Neighbourhoods Vs Crime Type Heatmap is displayed below:")
       neighbourhood_crime_type = df.groupby(['NEIGHBOURHOOD', 'TYPE']).size().reset_index().rename(columns={0: 'Crime_Count'})
       st.subheader("Neighbourhoods Vs Crime Type Heatmap")
       fig = px.density_heatmap(neighbourhood_crime_type, x='TYPE', y='NEIGHBOURHOOD', z='Crime_Count',
                    labels={'x': 'Crime Type', 'y': 'Neighbourhood', 'color': 'Number of Crimes'},
                    title='Neighbourhoods Vs Crime Type Heatmap')
       fig.update_layout(height=800, width=1000)
       st.plotly_chart(fig, width='content')

    elif visual_selection == "Neighbourhoods Crime Counts Yearly":
       st.write("Neighbourhoods Crime Count Yearly Bar Plot is displayed below:")
       neighbourhood_crime_count_yearly = df.groupby(['YEAR', 'NEIGHBOURHOOD']).size().reset_index().rename(columns={0: 'Crime_Count'})
       st.subheader("Yearly Neighbourhoods Crime Count")
       fig = px.bar(neighbourhood_crime_count_yearly, x='NEIGHBOURHOOD', y='Crime_Count', color='NEIGHBOURHOOD',
                 hover_data= {'YEAR': True, 'Crime_Count': True},
                 labels={'Crime_Count': 'Number of Crimes', 'NEIGHBOURHOOD': 'Neighbourhood'},
                 title='Neighbourhoods Crime Count Yearly')
       st.plotly_chart(fig)

    elif visual_selection == "Neighbourhoods Crime Counts Yearly Trend":
       st.write("Neighbourhoods Crime Count Yearly Trend Line Plot is displayed below:")
       neighbourhood_crime_type_yearly_trend = df.groupby(['YEAR', 'NEIGHBOURHOOD']).size().reset_index().rename(columns={0: 'Crime_Count'})
       st.subheader("Neighbourhoods Crime Count Yearly Trend")
       fig = px.line(neighbourhood_crime_type_yearly_trend, x='YEAR', y='Crime_Count', color='NEIGHBOURHOOD',
                 hover_data= {'YEAR': True, 'Crime_Count': True},
                 labels={'Crime_Count': 'Number of Crimes', 'YEAR': 'Year'},
                 title='Neighbourhoods Crime Count Yearly Trend')
       st.plotly_chart(fig)

    elif visual_selection == "Neighbourhoods Crime Type Counts Yearly":
       st.write("Neighbourhoods Crime Type Counts Yearly Bar Plot is displayed below:")
       neighbourhood_crime_type_count_yearly = df.groupby(['NEIGHBOURHOOD', 'TYPE', 'YEAR']).size().reset_index().rename(columns={0: 'Crime_Count'})
       st.subheader("Yearly Neighbourhoods Crime Type Counts")
       fig = px.bar(neighbourhood_crime_type_count_yearly, x='NEIGHBOURHOOD', y='Crime_Count', color='TYPE', facet_col='YEAR', facet_col_wrap=2,
                 hover_data= {'TYPE': True, 'YEAR': True, 'Crime_Count': True},
                 labels={'Crime_Count': 'Number of Crimes', 'NEIGHBOURHOOD': 'Neighbourhood'},
                 title='Neighbourhoods Crime Type Counts Yearly')
       fig.update_layout(height=1200, width=1200)
       st.plotly_chart(fig)

    elif visual_selection == "Neighbourhoods Crime Types Yearly Trend":
       st.write("Neighbourhoods Crime Type Counts Yearly Trend Line Plot is displayed below:")
       neighbourhood_crime_type_yearly_trend = df.groupby(['NEIGHBOURHOOD', 'TYPE', 'YEAR']).size().reset_index().rename(columns={0: 'Crime_Count'})
       st.subheader("Neighbourhoods Crime Type Counts Yearly Trend")
       fig = px.line(neighbourhood_crime_type_yearly_trend, x='YEAR', y='Crime_Count', color='NEIGHBOURHOOD', facet_col='TYPE', facet_col_wrap=3,
                 hover_data= {'TYPE': True, 'YEAR': True, 'Crime_Count': True},
                 labels={'Crime_Count': 'Number of Crimes', 'YEAR': 'Year'},
                 title='Neighbourhoods Crime Types Yearly Trend')
       st.plotly_chart(fig)

    elif visual_selection == "Neighbourhoods Crime Counts Monthly":
       st.write("Neighbourhoods Crime Count Monthly Bar Plot is displayed below:")
       neighbourhood_crime_count_monthly = df.groupby(['NEIGHBOURHOOD', 'YEAR', 'MONTH']).size().reset_index().rename(columns={0: 'Crime_Count'})
       st.subheader("Monthly Neighbourhoods Crime Count")
       fig = px.bar(neighbourhood_crime_count_monthly, x='NEIGHBOURHOOD', y='Crime_Count', color='NEIGHBOURHOOD', facet_col='MONTH', facet_col_wrap=4,
                 hover_data= {'MONTH': True, 'YEAR': True, 'Crime_Count': True},
                 labels={'Crime_Count': 'Number of Crimes', 'NEIGHBOURHOOD': 'Neighbourhood'},
                 title='Neighbourhoods Crime Count Monthly')
       st.plotly_chart(fig)

    elif visual_selection == "Neighbourhoods Crime Counts Monthly Trend":
       st.write("Neighbourhoods Crime Count Monthly Trend Line Plot is displayed below:")
       neighbourhood_crime_Count_monthly_trend = df.groupby(['YEAR', 'MONTH', 'NEIGHBOURHOOD']).size().reset_index().rename(columns={0: 'Crime_Count'})
       st.subheader("Neighbourhoods Crime Count Monthly Trend")
       fig = px.line(neighbourhood_crime_Count_monthly_trend, x='MONTH', y='Crime_Count', color='YEAR', facet_col='NEIGHBOURHOOD', facet_col_wrap=4,
                 hover_data= {'MONTH': True, 'YEAR': True, 'Crime_Count': True},
                 labels={'Crime_Count': 'Number of Crimes', 'MONTH': 'Month'},
                 title='Neighbourhoods Crime Count Monthly Trend')
       fig.update_layout(height=800, width=1200)
       st.plotly_chart(fig)

    elif visual_selection == "Neighbourhoods Crime Type Counts Monthly":
       st.write("Neighbourhoods Crime Type Counts Monthly Bar Plot is displayed below:")
       neighbourhood_crime_type_count_monthly = df.groupby(['NEIGHBOURHOOD', 'TYPE', 'YEAR', 'MONTH']).size().reset_index().rename(columns={0: 'Crime_Count'})
       st.subheader("Neighbourhoods Crime Type Counts Monthly")
       fig = px.bar(neighbourhood_crime_type_count_monthly, x='NEIGHBOURHOOD', y='Crime_Count', color='TYPE', facet_col='MONTH', facet_col_wrap=3,
                 hover_data= {'TYPE': True, 'MONTH': True, 'YEAR': True, 'Crime_Count': True},
                 labels={'Crime_Count': 'Number of Crimes', 'NEIGHBOURHOOD': 'Neighbourhood'},
                 title='Neighbourhoods Crime Type Counts Monthly')
       fig.update_layout(height=800, width=1200)
       st.plotly_chart(fig)

    elif visual_selection == "Neighbourhoods Crime Type Counts Monthly Trend":
       st.write("Neighbourhoods Crime Type Counts Monthly Trend Line Plot is displayed below:")
       neighbourhood_crime_type_monthly_trend = df.groupby(['NEIGHBOURHOOD', 'TYPE', 'YEAR', 'MONTH']).size().reset_index().rename(columns={0: 'Crime_Count'})
       st.subheader("Neighbourhoods Crime Type Counts Monthly Trend")
       fig = px.line(neighbourhood_crime_type_monthly_trend, x='MONTH', y='Crime_Count', color='NEIGHBOURHOOD', facet_col='TYPE', facet_col_wrap=3,
                 hover_data= {'TYPE': True, 'MONTH': True, 'YEAR': True, 'Crime_Count': True},
                 labels={'Crime_Count': 'Number of Crimes', 'MONTH': 'Month'},
                 title='Neighbourhoods Crime Type Counts Monthly Trend')
       fig.update_layout(height=800, width=1200)
       st.plotly_chart(fig)

    else:
       st.write("Please select a visualization option from the sidebar to display the corresponding visualizations.")                                          

# Conditional logic to display the appropriate dataset and visualizations based on user selection
elif dataset_selection == "LightGBM Crime Forecast":
    st.write("Current working directory:", os.getcwd())
    st.write("Files in root directory:", os.listdir())  
#load the historical data on which the LightGBM model was trained
    @st.cache_data
    def load_data():
        DATA_DIC = BASE_DIR / "Model Training Datasets"
        data_path = DATA_DIC / "Train_df_XGBoost_LightGBM_Models.csv"

        if not data_path.exists():
            st.error("Historical data file 'Train_df_XGBoost_LightGBM_Models.csv' not found. Please ensure the dataset file is in the correct path.")
            st.stop()

        df = pd.read_csv(data_path)
        df['YEAR'] = df['YEAR'].astype(int)
        df['MONTH'] = df['MONTH'].astype(int)
        df['ds'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str) + '-01')
        return df
    
#load the trained LightGBM model for crime forecasting
    st.write("Looking for model at:", BASE_DIR / "trained_LightGBM_forecast_model.pkl")
    st.write("Does file exist?", (BASE_DIR / "trained_LightGBM_forecast_model.pkl").exists())
    @st.cache_resource(show_spinner=False)
    def load_model(model_name):
      model_path = BASE_DIR / model_name

      if not model_path.exists():
        st.error(f"Trained LightGBM model file '{model_name}' not found. Please ensure the model file is in the correct path.")
        st.stop()

      return joblib.load(model_path)
    model = load_model("trained_LightGBM_forecast_model.pkl")
    st.success("Trained LightGBM model loaded successfully.")

    crime_type_df = load_data().copy()
    crime_type_df['YEAR'] = crime_type_df['YEAR'].astype(int)
    crime_type_df['MONTH'] = crime_type_df['MONTH'].astype(int)
    crime_type_df.rename(columns={'TYPE': 'unique_id', 'Monthly_Crime_Count_Type_wise': 'y'}, inplace=True)
    crime_type_df.sort_values(["unique_id", "YEAR", "MONTH"], inplace=True)
    crime_type_df.reset_index(drop=True, inplace=True)

# App title and description    
    st.title("LightGBM Regressor Deployment for Crime Forecasting")
    st.write("This section demonstrates the deployment of a trained LightGBM Regressor model for crime forecasting. The model has been trained on historical crime data and is now being used to predict future crime trends.")
    st.write("The dataset used for training the model includes various features such as crime type, year, month, and crime count. The model has been trained to capture the underlying patterns in the data and make accurate predictions for future crime counts based on these features.")
    st.write("Enter the features for crime forecasting in the sidebar and click the 'Predict' button to see the forecasted crime count based on the trained LightGBM model.")
# Create input fields for the features of historical data used in the model
    crime_types = sorted(crime_type_df['unique_id'].unique())
# year and month inputs are created with the same range as the historical data to check for historical data availability before using the model for prediction
    available_years = sorted([int(year) for year in crime_type_df['YEAR'].unique()], reverse=True)
# Allow users to pick a future year for prediction that is beyond the historical data range (eg., 2012 and 2013) to see the model's forecasting capability for future crime counts
    predict_years = list(range(min(available_years), max(available_years) + 3))  # Allow prediction for 2 years beyond the historical data

    selected_crime_type = st.sidebar.selectbox("Select Crime Type", options=crime_types)
    selected_year = st.sidebar.selectbox("Enter Year", predict_years, index=predict_years.index(datetime.now().year) if datetime.now().year in predict_years else 0)
    selected_month = st.sidebar.selectbox("Enter Month", list(range(1, 13)), index=datetime.now().month - 1)

# Logic to check for historical data and display results based on user input
    historical_match = crime_type_df[(crime_type_df['unique_id'] == selected_crime_type) & (crime_type_df['YEAR'] == selected_year) & (crime_type_df['MONTH'] == selected_month)]    
# Display results based on user input
    col1, col2 = st.columns(2)
# Check for historical data
# We filter the dataframe to check if the selected crime type, year, and month exist in the historical data
    with col1:
       st.subheader("Historical Data")
       if not historical_match.empty:
        actual_crime_count = historical_match['y'].values[0]
        st.metric("Actual Crime Count from Historical Data", int(actual_crime_count))
        st.info(f"Found record for {selected_month}/{selected_year}")
        st.dataframe(historical_match)
       else:
        st.metric(label="Historical Data Status", value="not available", delta="N/A", delta_color="off")
        st.warning("No historical data found for this timeframe.")

    with col2:
      st.subheader("Model Prediction")  
# If no historical data is found, we proceed to use the trained LightGBM model to predict the crime count for the selected crime type, year, and month
      if historical_match.empty:
        st.info(f"No historical data was found. So the model has forecasted the future crime count based on the user input.")

# Create a button to trigger the prediction        
        st.sidebar.button("Forecast", key="forecast_button")
        
        try:
            # Calculate months ahead from the last date in training data
            last_date = crime_type_df['ds'].max()
            selected_date = pd.Timestamp(f"{selected_year}-{selected_month}-01")
            months_ahead = (selected_date.year - last_date.year) * 12 + (selected_date.month - last_date.month) + 1
            
            # Build feature dataframe for ALL crime types (required by MLForecast)
            selected_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=months_ahead, freq='MS')
            all_crime_types = crime_type_df['unique_id'].unique()
            
            # Create a dataframe with all combinations of crime types and dates
            future_X_df = pd.DataFrame({
                'unique_id': [crime_type for crime_type in all_crime_types for _ in range(months_ahead)],
                'ds': list(selected_dates) * len(all_crime_types)
            })
            
            # Add all required features
            future_X_df['MONTH'] = future_X_df['ds'].dt.month
            future_X_df['is_summer'] = future_X_df['MONTH'].isin([5, 6, 7, 8]).astype(int)
            future_X_df['is_holiday_season'] = future_X_df['MONTH'].isin([9, 10, 11, 12]).astype(int)
            future_X_df['is_spring'] = future_X_df['MONTH'].isin([1, 2, 3, 4]).astype(int)
            future_X_df['quarter'] = ((future_X_df['MONTH'] - 1) // 3) + 1
            future_X_df['month_sin'] = np.sin(2 * np.pi * (future_X_df['MONTH'] - 1) / 12)
            future_X_df['month_cos'] = np.cos(2 * np.pi * (future_X_df['MONTH'] - 1) / 12)
            future_X_df['month_sq'] = future_X_df['MONTH'] ** 2
            future_X_df['summer_peak'] = future_X_df['is_summer'] * future_X_df['MONTH']
            future_X_df['holiday_peak'] = future_X_df['is_holiday_season'] * future_X_df['MONTH']
            
            # Generate forecast
            forecast_result = model.predict(h=months_ahead, X_df=future_X_df)
            
            # Get the prediction for the selected crime type and date
            mask = (forecast_result['unique_id'] == selected_crime_type) & (forecast_result['ds'] == selected_date)
            if not mask.any():
                st.error("Could not generate prediction for selected date.")
            else:
                predicted_crime_count = forecast_result[mask].iloc[-1, forecast_result.columns.get_loc('LGBMRegressor')]
                st.metric("Model Forecasted Crime Count", f"{predicted_crime_count:.2f}")
                st.caption("The predicted crime count is based on the trained LightGBM model using the input features provided.")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
      else:
        st.write("Historical data is available for the selected crime type, year, and month. The model prediction is not necessary as we have the actual crime count from the historical data.")

    st.divider()
          
# Showing the full forecasted dataset having only the forecasted crime counts of each month of the last 2 years (2012 and 2013) based on the trained LightGBM model
    st.subheader("Full Forecasted Dataset for 2012 and 2013 Based on Trained LightGBM Model")

    def generate_feature_columns_in_forecasted_dataset(crime_type_df):
# Create future dynamic features (X_df) for the next 24 months
        last_date = crime_type_df['ds'].max()
        future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=24, freq='MS')
        uids = crime_type_df['unique_id'].unique()

        X_df = pd.DataFrame({
       'unique_id': [i for i in uids for _ in range(24)],
       'ds': list(future_dates) * len(uids)})
        
        X_df['MONTH'] = X_df['ds'].dt.month
        X_df['is_summer'] = X_df['MONTH'].isin([5, 6, 7, 8]).astype(int)
        X_df['is_holiday_season'] = X_df['MONTH'].isin([9, 10, 11, 12]).astype(int)
        X_df['is_spring'] = X_df['MONTH'].isin([1, 2, 3, 4]).astype(int)
        X_df['quarter'] = ((X_df['MONTH'] - 1) // 3) + 1
        X_df['month_sin'] = np.sin(2 * np.pi * (X_df['MONTH'] - 1)/ 12)
        X_df['month_cos'] = np.cos(2 * np.pi * (X_df['MONTH'] - 1)/ 12)
        X_df['month_sq'] = X_df['MONTH'] ** 2
        X_df['summer_peak'] = X_df['is_summer'] * X_df['MONTH']
        X_df['holiday_peak'] = X_df['is_holiday_season'] * X_df['MONTH']

        return X_df
    X_df = generate_feature_columns_in_forecasted_dataset(crime_type_df)
# Final Forecast
    final_forecast1 = model.predict(h=24, X_df=X_df)
    st.subheader("Final Forecasted Crime Counts for Each Month of 2012 and 2013 Based on Trained LightGBM Model")
    print("\nFinal 24-Month Forecast:")
    print(final_forecast1.head())

# Visual representation of the forecasted crime counts for the last 2 years (2012 and 2013) based on the trained LightGBM model
# We create multiple bar charts(based on the unique_id) to visualize the forecasted monthly crime counts for each crime type over the last 2 years (2012 and 2013) based on the trained LightGBM model
    
    final_forecast2 = final_forecast1.copy()
    final_forecast2['YEAR'] = pd.to_datetime(final_forecast1['ds']).dt.year
    final_forecast2['MONTH'] = pd.to_datetime(final_forecast1['ds']).dt.month
    final_forecast2.rename(columns={'unique_id':'Crime_Type', 'LGBMRegressor': 'Crime_Count'}, inplace=True)

# Showing the forecasted crime counts for each month of the last 2 years (2012 and 2013) based on the trained LightGBM model in tabular format
    with st.expander("Click to view the full forecasted dataset for 2012 and 2013 based on the trained LightGBM model"):
      st.dataframe(final_forecast2[['Crime_Type', 'YEAR', 'MONTH', 'Crime_Count']])

# We create a bar plot to visualize the forecasted monthly crime counts for each crime type over the last 2 years (2012 and 2013) based on the trained LightGBM model    
    fig = px.bar(final_forecast2, x='YEAR', y='Crime_Count', color='MONTH', facet_col='Crime_Type',
                  hover_data= {'Crime_Type': True, 'YEAR': True, 'MONTH': True, 'Crime_Count': True},
                  labels={'YEAR': 'Year', 'Crime_Count': 'Forecasted Crime Count', 'Crime_Type': 'Crime Type'},
                  title='Forecasted Monthly Crime Counts for Each Crime Type (2012-2013)')
    st.plotly_chart(fig)

# We create multiple heatmaps to visualize the forecasted monthly crime counts for each crime type over the last 2 years (2012 and 2013) based on the trained LightGBM model
    fig = px.density_heatmap(final_forecast2, x='MONTH', y='YEAR', z='Crime_Count', facet_col='Crime_Type', facet_col_wrap=3,
                             hover_data= {'Crime_Type': True, 'YEAR': True, 'MONTH': True, 'Crime_Count': True},
                             labels={'MONTH': 'Month', 'YEAR': 'Year', 'Crime_Count': 'Forecasted Crime Count'},
                             title='Density Heatmap of Forecasted Monthly Crime Counts for Each Crime Type (2012-2013)')
    fig.update_layout(height=400, width=1200)
    fig.update_yaxes(type='category')
    fig.update_xaxes(type='category')
    st.plotly_chart(fig)

    st.divider()

    st.subheader("Comparison of Yearly Crime Trends of Each Crime Type Based on Past and Future Data")   

# Load the dataset containing the past and future crime counts predicted by the LightGBM model        
    def load_past_future_data():
      DATA_DIC = BASE_DIR / "VS Code Visualization Datasets"
      data_path2 = DATA_DIC / "Past_and_Future_df_type_wise_crime_count_LightBGM.csv"
      return pd.read_csv(data_path2)
    df1 = load_past_future_data().copy()
    df1['YEAR'] = df1['YEAR'].astype(int)
    df1['MONTH'] = df1['MONTH'].astype(int)
    st.write("Data Preview:")
    st.dataframe(df1.head(10))
    
# Yearly Crime Type Trend of Past and Future Data using Line Plot
    yearly_crime_count = df1.groupby(['TYPE', 'YEAR'])['Crime_Count'].sum().reset_index()
#    st.subheader("Yearly Crime Type Trend Based on Past and Future Data")
    fig = px.line(yearly_crime_count, x='YEAR', y='Crime_Count', color='TYPE',
                  labels={'YEAR': 'Year', 'Crime_Count': 'Number of Crimes'},
                  title='Yearly Crime Type Trend Based on Past and Future Data')
    st.plotly_chart(fig)

# Yearly Crime Trend of Each Crime Type Based on Past and Future Data using Multiple Line Plots
    yearly_crime_type_count = df1.groupby(['TYPE', 'YEAR', 'MONTH', 'Month_name'])['Crime_Count'].sum().reset_index()
    yearly_crime_type_count.sort_values(by=['TYPE', 'YEAR', 'MONTH', 'Month_name'], inplace=True)
    yearly_crime_type_count.reset_index(drop=True, inplace=True)
    fig = px.line(yearly_crime_type_count, x='Month_name', y='Crime_Count', color='YEAR', facet_col='TYPE', facet_col_wrap=3,
                  hover_data= {'TYPE': True, 'YEAR': True, 'Month_name': True, 'Crime_Count': True},
                  labels={'YEAR': 'Year', 'Month_name': 'Month', 'TYPE': 'Crime Type', 'Crime_Count': 'Number of Crimes'},
                  title='Yearly Crime Trend of Each Crime Type Based on Past and Future Data')
    st.plotly_chart(fig)

elif dataset_selection == "XGBoost Crime Forecast":
    st.write("Current working directory:", os.getcwd())
    st.write("Files in root directory:", os.listdir()) 
#load the historical data on which the XGBoost model was trained
    @st.cache_data
    def load_data():
        DATA_DIC = BASE_DIR / "Model Training Datasets"
        data_path = DATA_DIC / "Train_df_XGBoost_LightGBM_Models.csv"

        if not data_path.exists():
            st.error("Historical data file not found. Please ensure the data file is in the correct path.")
            st.stop()

        df = pd.read_csv(data_path)
        df['YEAR'] = df['YEAR'].astype(int)
        df['MONTH'] = df['MONTH'].astype(int)
        df['ds'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str) + '-01')
        return df
#load the trained XGBoost model for crime forecasting
    st.write("Looking for model at:", BASE_DIR / "trained_XGBoost_forecast_model.pkl")
    st.write("Does file exist?", (BASE_DIR / "trained_XGBoost_forecast_model.pkl").exists())
    @st.cache_resource(show_spinner=False)
    def load_model(model_filename):
        model_path = BASE_DIR / model_filename

        if not model_path.exists():
            st.error("Trained XGBoost model file not found. Please ensure the model file is in the correct path.")
            st.stop()

        return joblib.load(model_path)
    model = load_model("trained_XGBoost_forecast_model.pkl")
    st.success("Trained XGBoost model loaded successfully.")

    crime_type_df = load_data().copy()
    crime_type_df['YEAR'] = crime_type_df['YEAR'].astype(int)
    crime_type_df['MONTH'] = crime_type_df['MONTH'].astype(int)
    crime_type_df.rename(columns={'TYPE': 'unique_id', 'Monthly_Crime_Count_Type_wise': 'y'}, inplace=True)
    crime_type_df.sort_values(["unique_id", "YEAR", "MONTH"], inplace=True)
    crime_type_df.reset_index(drop=True, inplace=True)
    
# App title and description
    st.title("XGBoost Regressor Deployment for Crime Forecasting")
    st.write("This section demonstrates the deployment of a trained XGBoost Regressor model for crime forecasting. The model has been trained on historical crime data and is now being used to predict future crime trends.")
    st.write("The dataset used for training the model includes various features such as crime type, year, month, and crime count. The model has been trained to capture the underlying patterns in the data and make accurate predictions for future crime counts based on these features.")
    st.write("Enter the features for crime forecasting in the sidebar and click the 'Predict' button to see the forecasted crime count based on the trained XGBoost model.")
# Create input fields for the features of historical data used in the model
    crime_types = sorted(crime_type_df['unique_id'].unique())
# year and month inputs are created with the same range as the historical data to check for historical data availability before using the model for prediction
    available_years = sorted([int(year) for year in crime_type_df['YEAR'].unique()], reverse=True)
# Allow users to pick a future year for prediction that is beyond the historical data range (eg., 2012 and 2013) to see the model's forecasting capability for future crime counts
    predict_years = list(range(min(available_years), max(available_years) + 3))  # Allow prediction for 2 years beyond the historical data
    selected_crime_type = st.sidebar.selectbox("Select Crime Type", options=crime_types)
    selected_year = st.sidebar.selectbox("Enter Year", predict_years, index=predict_years.index(datetime.now().year) if datetime.now().year in predict_years else 0)
    selected_month = st.sidebar.selectbox("Enter Month", list(range(1, 13)), index=datetime.now().month - 1)

# Logic to check for historical data and display results based on user input
    historical_match = crime_type_df[(crime_type_df['unique_id'] == selected_crime_type) & (crime_type_df['YEAR'] == selected_year) & (crime_type_df['MONTH'] == selected_month)]    
# Display results based on user input
    col1, col2 = st.columns(2)
# Check for historical data
    with col1:
       st.subheader("Historical Data")
       if not historical_match.empty:
        actual_crime_count = historical_match['y'].values[0]
        st.metric("Actual Crime Count from Historical Data", int(actual_crime_count))
        st.info(f"Found record for {selected_month}/{selected_year}")
        st.dataframe(historical_match)
       else:
        st.metric(label="Historical Data Status", value="not available", delta="N/A", delta_color="off")
        st.warning("No historical data found for this timeframe.")
    with col2:
      st.subheader("Model Prediction")
      if historical_match.empty:
        st.info(f"No historical data was found. So the model has forecasted the future crime count based on the user input.")
        st.sidebar.button("Forecast", key="forecast_button")
        try:
            # Calculate months ahead from the last date in training data
            last_date = crime_type_df['ds'].max()
            selected_date = pd.Timestamp(f"{selected_year}-{selected_month}-01")
            months_ahead = (selected_date.year - last_date.year) * 12 + (selected_date.month - last_date.month) + 1
            
            # Build feature dataframe for ALL crime types (required by MLForecast)
            selected_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=months_ahead, freq='MS')
            all_crime_types = crime_type_df['unique_id'].unique()
            
            # Create a dataframe with all combinations of crime types and dates
            future_X_df = pd.DataFrame({
                'unique_id': [crime_type for crime_type in all_crime_types for _ in range(months_ahead)],
                'ds': list(selected_dates) * len(all_crime_types)
            })

            # Add all required features
            future_X_df['MONTH'] = future_X_df['ds'].dt.month
            future_X_df['is_summer'] = future_X_df['MONTH'].isin([5, 6, 7, 8]).astype(int)
            future_X_df['is_holiday_season'] = future_X_df['MONTH'].isin([9, 10, 11, 12]).astype(int)
            future_X_df['is_spring'] = future_X_df['MONTH'].isin([1, 2, 3, 4]).astype(int)
            future_X_df['quarter'] = ((future_X_df['MONTH'] - 1) // 3) + 1
            future_X_df['month_sin'] = np.sin(2 * np.pi * (future_X_df['MONTH'] - 1) / 12)
            future_X_df['month_cos'] = np.cos(2 * np.pi * (future_X_df['MONTH'] - 1) / 12)
            future_X_df['month_sq'] = future_X_df['MONTH'] ** 2
            future_X_df['summer_peak'] = future_X_df['is_summer'] * future_X_df['MONTH']
            future_X_df['holiday_peak'] = future_X_df['is_holiday_season'] * future_X_df['MONTH']
            
            # Generate forecast
            forecast_result = model.predict(h=months_ahead, X_df=future_X_df)
            # Get the prediction for the selected crime type and date
            mask = (forecast_result['unique_id'] == selected_crime_type) & (forecast_result['ds'] == selected_date)
            if not mask.any():
                st.error("Could not generate prediction for selected date.")
            else:                
               predicted_crime_count = forecast_result[mask].iloc[-1, forecast_result.columns.get_loc('XGBRegressor')]
               st.metric("Model Forecasted Crime Count", f"{predicted_crime_count:.2f}")
               st.caption("The predicted crime count is based on the trained XGBoost model using the input features provided.")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
      else:
         st.write("Historical data is available for the selected crime type, year, and month. The model prediction is not necessary as we have the actual crime count from the historical data.")

    st.divider()

# Showing the full forecasted dataset having only the forecasted crime counts of each month of the last 2 years (2012 and 2013) based on the trained XGBoost model
    st.subheader("Full Forecasted Dataset for 2012 and 2013 Based on Trained XGBoost Model")

    def generate_feature_columns_in_forecasted_dataset(crime_type_df):
# Create future dynamic features (X_df) for the next 24 months
        last_date = crime_type_df['ds'].max()
        future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=24, freq='MS')
        uids = crime_type_df['unique_id'].unique()

        X_df = pd.DataFrame({
       'unique_id': [i for i in uids for _ in range(24)],
       'ds': list(future_dates) * len(uids)})
        
        X_df['MONTH'] = X_df['ds'].dt.month
        X_df['is_summer'] = X_df['MONTH'].isin([5, 6, 7, 8]).astype(int)
        X_df['is_holiday_season'] = X_df['MONTH'].isin([9, 10, 11, 12]).astype(int)
        X_df['is_spring'] = X_df['MONTH'].isin([1, 2, 3, 4]).astype(int)
        X_df['quarter'] = ((X_df['MONTH'] - 1) // 3) + 1
        X_df['month_sin'] = np.sin(2 * np.pi * (X_df['MONTH'] - 1)/ 12)
        X_df['month_cos'] = np.cos(2 * np.pi * (X_df['MONTH'] - 1)/ 12)
        X_df['month_sq'] = X_df['MONTH'] ** 2
        X_df['summer_peak'] = X_df['is_summer'] * X_df['MONTH']
        X_df['holiday_peak'] = X_df['is_holiday_season'] * X_df['MONTH']

        return X_df
    X_df = generate_feature_columns_in_forecasted_dataset(crime_type_df)
# Final Forecast
    final_forecast1 = model.predict(h=24, X_df=X_df)
    st.subheader("Final Forecasted Crime Counts for Each Month of 2012 and 2013 Based on Trained XGBoost Model")
    print("\nFinal 24-Month Forecast:")
    print(final_forecast1.head())

# Visual representation of the forecasted crime counts for the last 2 years (2012 and 2013) based on the trained XGBoost model
# We create multiple bar charts(based on the unique_id) to visualize the forecasted monthly crime counts for each crime type over the last 2 years (2012 and 2013) based on the trained XGBoost model
    final_forecast2 = final_forecast1.copy()
    final_forecast2['YEAR'] = pd.to_datetime(final_forecast1['ds']).dt.year
    final_forecast2['MONTH'] = pd.to_datetime(final_forecast1['ds']).dt.month
    final_forecast2.rename(columns={'unique_id':'Crime_Type', 'XGBRegressor': 'Crime_Count'}, inplace=True)

# Showing the forecasted crime counts for each month of the last 2 years (2012 and 2013) based on the trained XGBoost model in tabular format
    with st.expander("Click to view the full forecasted dataset for 2012 and 2013 based on the trained XGBoost model"):
      st.dataframe(final_forecast2[['Crime_Type', 'YEAR', 'MONTH', 'Crime_Count']])
# We create a bar plot to visualize the forecasted monthly crime counts for each crime type over the last 2 years (2012 and 2013) based on the trained XGBoost model
    fig = px.bar(final_forecast2, x='YEAR', y='Crime_Count', color='MONTH', facet_col='Crime_Type',
                  hover_data= {'Crime_Type': True, 'YEAR': True, 'MONTH': True, 'Crime_Count': True},
                  labels={'YEAR': 'Year', 'Crime_Count': 'Forecasted Crime Count', 'Crime_Type': 'Crime Type'},
                  title='Forecasted Monthly Crime Counts for Each Crime Type (2012-2013)')
    st.plotly_chart(fig)
# We create multiple heatmaps to visualize the forecasted monthly crime counts for each crime type over the last 2 years (2012 and 2013) based on the trained XGBoost model
    fig = px.density_heatmap(final_forecast2, x='MONTH', y='YEAR', z='Crime_Count', facet_col='Crime_Type', facet_col_wrap=3,
                             hover_data= {'Crime_Type': True, 'YEAR': True, 'MONTH': True, 'Crime_Count': True},
                             labels={'MONTH': 'Month', 'YEAR': 'Year', 'Crime_Count': 'Forecasted Crime Count'},
                             title='Density Heatmap of Forecasted Monthly Crime Counts for Each Crime Type (2012-2013)')
    fig.update_layout(height=400, width=1200)
    fig.update_yaxes(type='category')
    fig.update_xaxes(type='category')
    st.plotly_chart(fig)

    st.divider()

    st.subheader("Comparison of Yearly Crime Trends of Each Crime Type Based on Past and Future Data")
# Load the dataset containing the past and future crime counts predicted by the XGBoost model            
    def load_past_future_data():
      DATA_DIC = BASE_DIR / "VS Code Visualization Datasets"
      data_path3 = DATA_DIC / "Past_and_Future_df_type_wise_crime_count_XGBoost.csv"
      return pd.read_csv(data_path3)
    df2 = load_past_future_data()
    df2['YEAR'] = df2['YEAR'].astype(int)
    df2['MONTH'] = df2['MONTH'].astype(int)
    st.write("Crime Count Forecast using XGBoost Regressor Model:")
    st.write("Data Preview:")
    st.dataframe(df2.head(10))

# Yearly Crime Trend of Past and Future Data using Line Plot
    yearly_crime_count = df2.groupby(['TYPE', 'YEAR'])['Crime_Count'].sum().reset_index()
#    st.subheader("Yearly Crime Type Trend Based on Past and Future Data")
    fig = px.line(yearly_crime_count, x='YEAR', y='Crime_Count', color='TYPE',
                  labels={'YEAR': 'Year', 'Crime_Count': 'Number of Crimes'},
                  title='Yearly Crime Type Trend Based on Past and Future Data')
    st.plotly_chart(fig)
# Yearly Crime Trend of Each Crime Type Based on Past and Future Data using Multiple Line Plots
    yearly_crime_type_count = df2.groupby(['TYPE', 'YEAR', 'MONTH', 'Month_name'])['Crime_Count'].sum().reset_index()
    yearly_crime_type_count.sort_values(by=['TYPE', 'YEAR', 'MONTH', 'Month_name'], inplace=True)
    yearly_crime_type_count.reset_index(drop=True, inplace=True)
    fig = px.line(yearly_crime_type_count, x='Month_name', y='Crime_Count', color='YEAR', facet_col='TYPE', facet_col_wrap=3,
                  hover_data= {'TYPE': True, 'YEAR': True, 'Month_name': True, 'Crime_Count': True},
                  labels={'YEAR': 'Year', 'Month_name': 'Month', 'TYPE': 'Crime Type', 'Crime_Count': 'Number of Crimes'},
                  title='Yearly Crime Trend of Each Crime Type Based on Past and Future Data')
    st.plotly_chart(fig)
elif dataset_selection == "SARIMAX Crime Forecast":
    st.write("Current working directory:", os.getcwd())
    st.write("Files in root directory:", os.listdir()) 
# Load the dataset in which the SARIMAX model was trained
    @st.cache_data
    def load_data():
        DATA_DIC = BASE_DIR / "Model Training Datasets"
        data_path = DATA_DIC / "Train_df_SARIMAX_Model.csv"

        if not data_path.exists():
          st.error(f"Dataset not found: {data_path}")
          st.stop()

        df = pd.read_csv(data_path)
        df['YEAR'] = df['YEAR'].astype(int)
        df['ds'] = pd.to_datetime(df['YEAR'].astype(str) + '-01-01')

        return df
    
# Load the SARIMAX model for crime forecasting
    st.write("Looking for model at:", BASE_DIR / "final_SARIMA_forecast.pkl")
    st.write("Does file exist?", (BASE_DIR / "final_SARIMA_forecast.pkl").exists())
    @st.cache_resource(show_spinner=False)
    def load_model(model_filename):
        model_path = BASE_DIR / model_filename

        if not model_path.exists():
            st.error("Trained SARIMAX model file not found. Please ensure the model file is in the correct path.")
            st.stop()

        return joblib.load(model_path)
    
    model = load_model("final_SARIMA_forecast.pkl")
    st.write("Trained SARIMAX model loaded successfully.")

    crime_neighbourhood_df = load_data().copy()
    crime_neighbourhood_df['YEAR'] = crime_neighbourhood_df['YEAR'].astype(int)
    crime_neighbourhood_df.rename(columns={'NEIGHBOURHOOD': 'unique_id', 'Total_Yearly_Crime_Count_per_Neighbourhood': 'y'}, inplace=True)
    crime_neighbourhood_df.sort_values(["unique_id", "YEAR"], inplace=True)
    crime_neighbourhood_df.reset_index(drop=True, inplace=True)

# App title and description
    st.title("SARIMAX Model Deployment for Crime Forecasting")
    st.write("This section demonstrates the deployment of a trained SARIMAX model for crime forecasting. The model has been trained on historical crime data and is now being used to predict future crime trends.")
    st.write("The dataset used for training the model includes various features such as neighbourhood, year, and crime count. The model has been trained to capture the underlying patterns in the data and make accurate predictions for future crime counts based on these features.")
    st.write("Enter the features for crime forecasting in the sidebar and click the 'Predict' button to see the forecasted crime count based on the trained SARIMAX model.")

# Create input fields for the features of historical data used in the model
    neighbourhoods = sorted(crime_neighbourhood_df['unique_id'].unique())
# year input is created with the same range as the historical data to check for historical data availability before using the model for prediction
    available_years = sorted([int(year) for year in crime_neighbourhood_df['YEAR'].unique()], reverse=True)
# Allow users to pick a future year for prediction that is beyond the historical data range (eg., 2012 and 2013) to see the model's forecasting capability for future crime counts
    predict_years = list(range(min(available_years), max(available_years) + 3))  # Allow prediction for 2 years beyond the historical data
    selected_neighbourhood = st.sidebar.selectbox("Select Neighbourhood", options=neighbourhoods)
    selected_year = st.sidebar.selectbox("Enter Year", predict_years, index=predict_years.index(datetime.now().year) if datetime.now().year in predict_years else 0)

# Logic to check for historical data and display results based on user input
    historical_match = crime_neighbourhood_df[(crime_neighbourhood_df['unique_id'] == selected_neighbourhood) & (crime_neighbourhood_df['YEAR'] == selected_year)]    
# Display results based on user input
    col1, col2 = st.columns(2)
# Check for historical data
    with col1:
       st.subheader("Historical Data")
       if not historical_match.empty:
        actual_crime_count = historical_match['y'].values[0]
        st.metric("Actual Crime Count from Historical Data", int(actual_crime_count))
        st.info(f"Found record for {selected_year}")
        st.dataframe(historical_match)
       else:
        st.metric(label="Historical Data Status", value="not available", delta="N/A", delta_color="off")
        st.warning("No historical data found for this timeframe.")
    with col2:
        st.subheader("Model Prediction")
        if historical_match.empty:
            st.info(f"No historical data was found. So the model has forecasted the future crime count based on the user input.")
            st.sidebar.button("Forecast", key="forecast_button")
            try:
                # Calculate years ahead from the last date in training data
                last_year = crime_neighbourhood_df['YEAR'].max()
                selected_date = pd.Timestamp(f"{selected_year}-01-01")
                years_ahead = selected_date.year - last_year

                # Validate forecast horizon
                if years_ahead <= 0:
                    st.error(f"Selected year {selected_year} is not after the last training year {last_year}. Choose a later year for forecasting.")
                else:
                    # Build feature dataframe for ALL neighbourhoods (required by some forecasting logic)
                    selected_dates = pd.date_range(start=pd.Timestamp(f"{last_year + 1}-01-01"), periods=years_ahead, freq='YS')
                    all_neighbourhoods = crime_neighbourhood_df['unique_id'].unique()

                    # Create a dataframe with all combinations of neighbourhoods and dates (not directly used by saved function but kept for compatibility)
                    future_X_df = pd.DataFrame({
                        'unique_id': [neighbourhood for neighbourhood in all_neighbourhoods for _ in range(years_ahead)],
                        'ds': list(selected_dates) * len(all_neighbourhoods)
                    })

                    # Ensure years_ahead is an integer >= 1
                    try:
                        years_ahead = int(years_ahead)
                    except Exception:
                        st.error("Internal error: invalid forecast horizon.")
                        raise

                    # Filter out neighbourhoods with no observations to avoid SARIMAX fitting errors
                    valid_uids = []
                    for uid in crime_neighbourhood_df['unique_id'].unique():
                        cnt = crime_neighbourhood_df[crime_neighbourhood_df['unique_id'] == uid]['y'].dropna().shape[0]
                        if cnt > 0:
                            valid_uids.append(uid)

                    if len(valid_uids) == 0:
                        st.error("No neighbourhood time series have observations to forecast.")
                        raise RuntimeError("No valid series for forecasting")

                    filtered_df = crime_neighbourhood_df[crime_neighbourhood_df['unique_id'].isin(valid_uids)].copy()

                    # Generate forecast using the saved forecasting function and handle errors
                    try:
                        forecast_result = model(filtered_df, forecast_years=years_ahead)
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
                        raise

                    # Get the prediction for the selected neighbourhood and date
                    mask = (forecast_result['unique_id'] == selected_neighbourhood) & (forecast_result['ds'] == selected_date)
                    if not mask.any():
                        st.error("Could not generate prediction for selected date.")
                    else:
                        predicted_crime_count = float(forecast_result.loc[mask, 'prediction'].iloc[-1])
                        st.metric("Model Forecasted Crime Count", f"{predicted_crime_count:.2f}")
                        st.caption("The predicted crime count is based on the trained SARIMAX model using the input features provided.")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
        else:
             st.write("Historical data is available for the selected neighbourhood and year. The model prediction is not necessary as we have the actual crime count from the historical data.")

    st.divider()

# Showing the full forecasted dataset having only the forecasted crime counts of each year of the last 2 years (2012 and 2013) based on the trained SARIMAX model
    st.subheader("Full Forecasted Dataset for 2012 and 2013 Based on Trained SARIMAX Model")

    def generate_feature_columns_in_forecasted_dataset(crime_neighbourhood_df):
# Create future dynamic features (X_df) for the next 2 years
        last_year = crime_neighbourhood_df['YEAR'].max()
        future_dates = pd.date_range(start=pd.Timestamp(f"{last_year + 1}-01-01"), periods=2, freq='YS')
        uids = crime_neighbourhood_df['unique_id'].unique()

        X_df = pd.DataFrame({
       'unique_id': [i for i in uids for _ in range(2)],
       'ds': list(future_dates) * len(uids)})
        
        X_df['YEAR'] = X_df['ds'].dt.year

        return X_df
    X_df = generate_feature_columns_in_forecasted_dataset(crime_neighbourhood_df)
# Final Forecast
    # Use the saved forecasting function to generate the final forecast
    final_forecast1 = model(crime_neighbourhood_df, forecast_years=2)
    # Normalize column name to match existing downstream code
    if 'prediction' in final_forecast1.columns:
        final_forecast1 = final_forecast1.rename(columns={'prediction': 'SARIMAX'})

    st.subheader("Final Forecasted Crime Counts for Each Year of 2012 and 2013 Based on Trained SARIMAX Model")
    print("\nFinal 2-Year Forecast:")
    print(final_forecast1.head())

# Visual representation of the forecasted crime counts for each neighbourhood for the last 2 years (2012 and 2013) based on the trained SARIMAX model
# We create multiple line plots to visualize the forecasted yearly crime counts for each neighbourhood for the last 2 years (2012 and 2013) based on the trained SARIMAX model
# We also create a heatmap to visualize the forecasted yearly crime counts for each neighbourhood for the last 2 years (2012 and 2013) based on the trained SARIMAX model
    final_forecast2 = final_forecast1.copy()
    final_forecast2['YEAR'] = pd.to_datetime(final_forecast2['ds']).dt.year
    final_forecast2.rename(columns={'unique_id':'NEIGHBOURHOOD', 'SARIMAX': 'Crime_Count'}, inplace=True)

# Showing the forecasted crime counts for each year of the last 2 years (2012 and 2013) based on the trained SARIMAX model in tabular format
    with st.expander("Click to view the full forecasted dataset for 2012 and 2013 based on the trained SARIMAX model"):
      st.dataframe(final_forecast2[['NEIGHBOURHOOD', 'YEAR', 'Crime_Count']])
# Showing Forecasted Yearly Crime Count of Each Neighbourhood using Bar Plot
    yearly_crime_count_neighbourhood = final_forecast2.groupby(['NEIGHBOURHOOD', 'YEAR'])['Crime_Count'].sum().reset_index()
    yearly_crime_count_neighbourhood.sort_values(by=['NEIGHBOURHOOD', 'YEAR'], inplace=True)
    yearly_crime_count_neighbourhood.reset_index(drop=True, inplace=True)
    fig = px.bar(yearly_crime_count_neighbourhood, x='YEAR', y='Crime_Count', color='NEIGHBOURHOOD',
                  hover_data= {'NEIGHBOURHOOD': True, 'YEAR': True, 'Crime_Count': True},
                  labels={'YEAR': 'Year', 'NEIGHBOURHOOD': 'Neighbourhood', 'Crime_Count': 'Number of Crimes'},
                  title='Yearly Crime Count of Each Neighbourhood Based on Past and Future Data')
    st.plotly_chart(fig)
# Showing Percentage Share Forecasted Yearly Crime Count of Each Neighbourhood using Pie Charts
    fig = px.pie(yearly_crime_count_neighbourhood, values='Crime_Count', names='NEIGHBOURHOOD', facet_col='YEAR', hole=0.4,
                 hover_data= {'NEIGHBOURHOOD': True, 'YEAR': True, 'Crime_Count': True},
                 labels={'NEIGHBOURHOOD': 'Neighbourhood', 'Crime_Count': 'Number of Crimes', 'YEAR': 'Year'},
                 title='Percentage Share of Yearly Crime Count of Each Neighbourhood Based on Past and Future Data')
    fig.update_layout(height=550, width=900)
    st.plotly_chart(fig)

    st.divider()

# Load the dataset containing the past and future crime counts predicted by the SARIMAX model
    st.subheader("Comparison of Yearly Crime Trends of Each Neighbourhood Based on Past and Future Data")        
    data_path4 = "VS Code Visualization Datasets/Past_and_Future_df_yearly_neighbourhood_crime_count_sarimax.csv"
    df3 = pd.read_csv(data_path4)
    df3['YEAR'] = df3['YEAR'].astype(int)
    st.write("Crime Count per Neighbourhood Forecast using SARIMAX Model:")
    st.write("Data Preview:")
    st.dataframe(df3.head(10))

# Yearly Crime Count Trend of Each Neighbourhood Based on Past and Future Data using Multiple Line Plots
    yearly_crime_count_neighbourhood = df3.groupby(['NEIGHBOURHOOD', 'YEAR'])['Crime_Count'].sum().reset_index()
    yearly_crime_count_neighbourhood.sort_values(by=['NEIGHBOURHOOD', 'YEAR'], inplace=True)
    yearly_crime_count_neighbourhood.reset_index(drop=True, inplace=True)
    fig = px.line(yearly_crime_count_neighbourhood, x='YEAR', y='Crime_Count', color='NEIGHBOURHOOD',
                  hover_data= {'NEIGHBOURHOOD': True, 'YEAR': True, 'Crime_Count': True},
                  labels={'YEAR': 'Year', 'NEIGHBOURHOOD': 'Neighbourhood', 'Crime_Count': 'Number of Crimes'},
                  title='Yearly Crime Count Trend of Each Neighbourhood Based on Past and Future Data')
    st.plotly_chart(fig)
# Yearly Crime Count Trend of Each Neighbourhood Based on Past and Future Data using Heatmap
    yearly_crime_count_neighbourhood_heatmap = df3.pivot_table(index='NEIGHBOURHOOD', columns='YEAR', values='Crime_Count', aggfunc='sum', fill_value=0)
#    st.subheader("Yearly Crime Count Trend of Each Neighbourhood Based on Past and Future Data Heatmap")
    fig = px.imshow(yearly_crime_count_neighbourhood_heatmap,
                    labels=dict(x="Year", y="Neighbourhood", color="Crime Count"),
                    title="Yearly Crime Count Trend of Each Neighbourhood Based on Past and Future Data Heatmap")
    st.plotly_chart(fig)
else:
    st.write("Please upload a CSV file to visualize the data.")
