import streamlit as st
import datetime
import requests
import pandas as pd
import pytz
import joblib
import os

'''
# Taxi Fare Prediction
'''

# Creating boxes for user input
columns = st.columns(3)
date = columns[0].date_input("Enter date", datetime.date(2019, 7, 6))
time = columns[1].time_input('Enter time', datetime.time(8, 45))
date_and_time = datetime.datetime.combine(date, time)
passenger_count = columns[2].selectbox('Number of passengers', list(range(1, 9)))

columns = st.columns(2)
pickup_longitude = columns[0].number_input('Enter pickup longitude', value=-73.950655, format="%.5f")
pickup_latitude = columns[1].number_input('Enter pickup latitude', value=40.783282, format="%.5f")

columns = st.columns(2)
dropoff_longitude = columns[0].number_input('Enter dropoff longitude', value=-73.984365, format="%.5f")
dropoff_latitude = columns[1].number_input('Enter dropoff latitude', value=40.769802, format="%.5f")


col1, col2, col3 = st.columns(3)
predict_button = col2.button('Predict Taxi Fare')

params=dict(pickup_datetime=date_and_time,
            pickup_longitude=pickup_longitude,
            pickup_latitude=pickup_latitude,
            dropoff_longitude=dropoff_longitude,
            dropoff_latitude=dropoff_latitude,
            passenger_count=int(passenger_count))

# Geocode location
geocode_url = 'https://nominatim.openstreetmap.org/reverse'
geocode_pickup = requests.get(geocode_url, params={'lat':pickup_latitude,
                                                   'lon':pickup_longitude,
                                                   'format':'json'}).json()
geocode_dropoff = requests.get(geocode_url, params={'lat':dropoff_latitude,
                                                    'lon':dropoff_longitude,
                                                    'format':'json'}).json()
st.markdown(f"##### _Pickup location: {geocode_pickup['display_name']}_")
st.markdown(f"##### _Dropoff location: {geocode_dropoff['display_name']}_")

# Plot coordinates on a map
coord_df = pd.DataFrame({'lat': [pickup_latitude, dropoff_latitude],
                         'lon': [pickup_longitude, dropoff_longitude]})
st.map(coord_df)

if predict_button:
    ############################################################################
    # Loading model directly
    ############################################################################

    ## Modifying the format of params['pickup_datetime'] as per requirements in the model
    # localize the user provided datetime with the NYC timezone
    eastern = pytz.timezone("US/Eastern")
    localized_pickup_datetime = eastern.localize(params['pickup_datetime'], is_dst=None)
    # convert the user datetime to UTC
    utc_pickup_datetime = localized_pickup_datetime.astimezone(pytz.utc)
    # format the datetime as expected by the pipeline
    params['pickup_datetime'] = utc_pickup_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")

    # Fixing a value for the key, as the pipeline was created with the original dataset
    # which contains a key column. This column was dropped later on in the pipeline
    # and not used in training the model.
    key='2013-07-06 17:18:00.000000119'
    X_pred = pd.DataFrame(params, index=[0])
    X_pred.insert(loc=0, column='key', value=key)
    model = joblib.load(os.path.dirname(__file__) + "/../joblib/model.joblib")
    fare = float(model.predict(X_pred).round(2))
    st.markdown(f'## Predicted fare by own model: `{fare}`')


    ############################################################################
    # Using API
    ############################################################################
    # # # If using own API, will first need to create a heroku website taxifareapi-marcus
    # # # modify Procfile to web: sh setup.sh && uvicorn app.api:api --host=0.0.0.0 --port=$PORT
    # # # and git push origin/heroku master (origin works as github has been configured
    # # # to deploy to Heroku as well)
    # # url = 'https://taxifareapi-marcus.herokuapp.com/predict'

    # # Le wagon API
    # url = 'https://taxifare.lewagon.ai/predict'

    # # Setting params['pickup_datetime'] to corect format for API
    # params['pickup_datetime'] = date_and_time
    # response = requests.get(url, params=params).json()
    # fare = round(response['fare'], 2)
    # st.markdown(f'## Predicted fare by Le Wagon API: `{fare}`')
