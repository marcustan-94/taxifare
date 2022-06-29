from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from datetime import datetime
import pytz
import joblib
import os


api = FastAPI()

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# define a root `/` endpoint
@api.get("/")
def index():
    return {"Taxi fare prediction API": "Hello"}


@api.get("/predict")
def predict(pickup_datetime,        # 2013-07-06 17:18:00
            pickup_longitude,       # -73.950655
            pickup_latitude,        # 40.783282
            dropoff_longitude,      # -73.984365
            dropoff_latitude,       # 40.769802
            passenger_count):       # 1


    # create datetime object from user provided date
    pickup_datetime = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")
    # localize the user provided datetime with the NYC timezone
    eastern = pytz.timezone("US/Eastern")
    localized_pickup_datetime = eastern.localize(pickup_datetime, is_dst=None)
    # convert the user datetime to UTC
    utc_pickup_datetime = localized_pickup_datetime.astimezone(pytz.utc)
    # format the datetime as expected by the pipeline
    formatted_pickup_datetime = utc_pickup_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")

    # fixing a value for the key, unused by the model
    key='2013-07-06 17:18:00.000000119'

    params=dict(key=[key],
                pickup_datetime=[formatted_pickup_datetime],
                pickup_longitude=[float(pickup_longitude)],
                pickup_latitude=[float(pickup_latitude)],
                dropoff_longitude=[float(dropoff_longitude)],
                dropoff_latitude=[float(dropoff_latitude)],
                passenger_count=[int(passenger_count)]
                )
    X_pred = pd.DataFrame(params)
    model = joblib.load(os.path.dirname(__file__) + "/../joblib/model.joblib")
    fare = float(model.predict(X_pred).round(2))
    return {'fare': fare}

if __name__ == '__main__':
    test_fare = predict('2013-07-06 17:18:00',
                        '-73.950655',
                        '40.783282',
                        '-73.984365',
                        '40.769802',
                        '1')
    print(test_fare)


# Test code
# http://127.0.0.1:8000/predict?pickup_datetime=2013-07-06 17:18:00&pickup_longitude=-73.950655&pickup_latitude=40.783282&dropoff_longitude=-73.984365&dropoff_latitude=40.769802&passenger_count=1
