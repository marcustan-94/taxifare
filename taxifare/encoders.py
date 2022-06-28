import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from taxifare.data import get_data, clean_df
from taxifare.utils import haversine_vectorized

class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """Extract the day of week (dow), the hour, the month and the year from a time column.
    Returns a copy of the DataFrame X with only four columns: 'dow', 'hour', 'month', 'year'.
    """

    def __init__(self, time_column='pickup_datetime', time_zone_name='America/New_York'):
        self.time_column = time_column
        self.time_zone_name = time_zone_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X.index = pd.to_datetime(X[self.time_column])
        X.index = X.index.tz_convert(self.time_zone_name)
        X["dow"] = X.index.weekday
        X["hour"] = X.index.hour
        X["month"] = X.index.month
        X["year"] = X.index.year
        return X[['dow', 'hour', 'month', 'year']]


class DistanceTransformer(BaseEstimator, TransformerMixin):
    """Compute the haversine distance between two GPS points.
    Returns a copy of the DataFrame X with only one column: 'distance', units km.
    """
    def __init__(self,
                 start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude"):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X["distance"] = haversine_vectorized(
            X,
            start_lat=self.start_lat,
            start_lon=self.start_lon,
            end_lat=self.end_lat,
            end_lon=self.end_lon
        )
        return X[['distance']]


class DistanceToCenter(BaseEstimator, TransformerMixin):
    """Compute the haversine distance to the NYC center for both pickup and dropoff locations."""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        nyc_center = (40.7141667, -74.0063889)
        X["nyc_lat"], X["nyc_lng"] = nyc_center[0], nyc_center[1]
        kwargs_pickup = dict(start_lat="nyc_lat", start_lon="nyc_lng",
                           end_lat="pickup_latitude", end_lon="pickup_longitude")
        kwargs_dropoff = dict(start_lat="nyc_lat", start_lon="nyc_lng",
                            end_lat="dropoff_latitude", end_lon="dropoff_longitude")
        X['pickup_distance_to_center'] = haversine_vectorized(X, **kwargs_pickup)
        X['dropoff_distance_to_center'] = haversine_vectorized(X, **kwargs_dropoff)
        return X[["pickup_distance_to_center", "dropoff_distance_to_center"]]


if __name__ == "__main__":
    df = get_data()
    df = clean_df(df)

    time_features = TimeFeaturesEncoder()
    time_features_df = time_features.transform(df)
    print(time_features_df)

    distance_transformer = DistanceTransformer()
    distance_transfomer_df = distance_transformer.transform(df)
    print(distance_transfomer_df)

    dist_to_center = DistanceToCenter()
    dist_to_center_df = dist_to_center.transform(df)
    print(dist_to_center_df)
