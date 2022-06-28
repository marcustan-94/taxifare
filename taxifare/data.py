import os

import pandas as pd

LOCAL_PATH = os.path.join(os.path.dirname(__file__), '..', 'raw_data', 'train.csv')

def get_data(nrows=10_000, **kwargs):
    '''returns a DataFrame with nrows'''
    df = pd.read_csv(LOCAL_PATH, nrows=nrows)
    return df


def clean_df(df):
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    df = df[(df['pickup_latitude'] != df['dropoff_latitude']) & \
            (df['pickup_longitude'] != df['dropoff_longitude'])]
    if "fare_amount" in list(df):
        df = df[df['fare_amount'].between(0, 4000, inclusive='right')]
    df = df[df.passenger_count <= 8]
    df = df[df.passenger_count > 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df
