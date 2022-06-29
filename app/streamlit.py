import streamlit as st
import datetime
import requests
import pandas as pd

'''
# Taxi Fare Prediction
'''

columns = st.columns(2)
date = columns[0].date_input("Enter date", datetime.date(2019, 7, 6))
time = columns[1].time_input('Enter time', datetime.time(8, 45))
date_and_time = datetime.datetime.combine(date, time)

columns = st.columns(2)
pickup_longitude = columns[0].number_input('Enter pickup longitude', format="%.5f")
pickup_latitude = columns[1].number_input('Enter pickup latitude', format="%.5f")

columns = st.columns(2)
dropoff_longitude = columns[0].number_input('Enter dropoff longitude', format="%.5f")
dropoff_latitude = columns[1].number_input('Enter dropoff latitude', format="%.5f")
passenger_count = st.selectbox('Number of passengers', list(range(1, 9)))


geocode_url = 'https://nominatim.openstreetmap.org/reverse'

geocode_pickup = requests.get(geocode_url, params={'lat':pickup_latitude,
                                                   'lon':pickup_longitude,
                                                   'format':'json'}).json()
geocode_dropoff = requests.get(geocode_url, params={'lat':dropoff_latitude,
                                                    'lon':dropoff_longitude,
                                                    'format':'json'}).json()

st.markdown(f"##### _Pickup location: {geocode_pickup['display_name']}_")
st.markdown(f"##### _Dropoff location: {geocode_dropoff['display_name']}_")

params=dict(pickup_datetime=date_and_time,
            pickup_longitude=pickup_longitude,
            pickup_latitude=pickup_latitude,
            dropoff_longitude=dropoff_longitude,
            dropoff_latitude=dropoff_latitude,
            passenger_count=int(passenger_count))

@st.cache
def get_map_data():

    df = pd.DataFrame({'lat': [pickup_latitude, dropoff_latitude],
                       'lon': [pickup_longitude, dropoff_longitude]
                       })

    return df


df = get_map_data()
st.map(df)

url = 'https://taxifare.lewagon.ai/predict'
response = requests.get(url, params=params).json()
fare = round(response['fare'], 2)

st.markdown(f'## Predicted fare: `{fare}`')
