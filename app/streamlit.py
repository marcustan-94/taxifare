import datetime
import os

import joblib
import pandas as pd
import pytz
import requests
import streamlit as st

'''
# Taxi Fare Predictor (New York City)
'''
CSS = """
.css-fg4pbf {background: #FFFFE8;}
.st-br {background-color: white;}
"""
st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)


def address_coord(entered_address):
    api_url = 'https://nominatim.openstreetmap.org/search'
    response = requests.get(api_url, params={'q':entered_address,
                                                'format':'json'}).json()

    for address in response:
        if address['display_name'].lower() == entered_address:
            lon = float(address['lon'])
            lat = float(address['lat'])
            return lon, lat

    if len(response) == 1:
        st.markdown("*We found one possible address but it doesn't match your query exactly.\
                Please refine your search or copy this address into the query box.*")
        st.markdown(f"""`{address['display_name']}`""")
    elif len(response) > 1:
        st.write("*We found multiple possible addresses. Please refine your search or\
                copy one of the addresses below into the query box:*")
        for index, address in enumerate(response[:3]):
            st.markdown(f"""`{index + 1})  {address['display_name']}`""")
    elif len(response) == 0 and entered_address != "":
        st.write("*No addresses were found. Please refine your search.*")

    return -9999, -9999


# Creating boxes for user input
columns = st.columns(3)
date = columns[0].date_input("Enter date", datetime.date(2019, 7, 6))
time = columns[1].time_input('Enter time', datetime.time(8, 45))
date_and_time = datetime.datetime.combine(date, time)
passenger_count = columns[2].selectbox('Number of passengers', list(range(1, 9)))

# Obtaining coordinates for pickup and dropoff address
pickup_address = st.text_input('Enter pickup address').strip().lower()
pickup_longitude, pickup_latitude = address_coord(pickup_address)
dropoff_address = st.text_input('Enter dropoff address').strip().lower()
dropoff_longitude, dropoff_latitude = address_coord(dropoff_address)

col1, col2, col3 = st.columns(3)
predict_button = col2.button('Predict Taxi Fare')

if predict_button:
    if pickup_address == "" or dropoff_address == "":
        st.write("**Error! Please enter a valid address for both boxes.**")
    elif pickup_longitude + pickup_latitude + dropoff_longitude + dropoff_latitude < -5000:
        st.write("**Error! The address(es) entered is/are invalid.**")
    elif pickup_longitude == dropoff_longitude and pickup_latitude == dropoff_latitude:
        st.write("**Error! Please make sure that the pickup and dropoff locations are different.**")


    else:
        params=dict(pickup_datetime=date_and_time,
                    pickup_longitude=pickup_longitude,
                    pickup_latitude=pickup_latitude,
                    dropoff_longitude=dropoff_longitude,
                    dropoff_latitude=dropoff_latitude,
                    passenger_count=int(passenger_count))

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
        st.markdown(f'## Predicted fare: `{fare}`')

        # Plot coordinates on a map
        coord_df = pd.DataFrame({'lat': [pickup_latitude, dropoff_latitude],
                                'lon': [pickup_longitude, dropoff_longitude]})
        st.map(coord_df)

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
    # st.markdown(f'## Predicted fare: `{fare}`')

disclaimer = st.markdown("""
                ### **Disclaimer:**

                Due to memory constraints of Heroku, the model loaded here is not
                the optimized version as it has only been trained on a very small subset
                of the dataset; hence this website is purely for demonstration purposes only.

                While the website accepts addresses outside of New York city, the model
                likely will not be able to give an accurate prediction, as the model
                was trained using data from the Kaggle New York Taxi Fare dataset.
                """)
