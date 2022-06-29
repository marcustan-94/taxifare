import joblib
import mlflow
import pandas as pd
from taxifare.data import get_data, clean_df
from taxifare.encoders import TimeFeaturesEncoder, DistanceTransformer, DistanceToCenter
from taxifare.utils import compute_rmse
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MLFLOW_URI = "https://mlflow.lewagon.ai/"
EXPERIMENT_NAME = '[SG] [Singapore] [marcustan-94] taxifare v0'

class Trainer():
    def __init__(self, X, y, **kwargs):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.kwargs = kwargs
        self.model = kwargs.get('model', None)
        self.model_name = str(self.model).lower()[:str(self.model).find('(')]
        self.model_params = str(self.model)[str(self.model).find('('):]
        self.dist_encoder = kwargs.get('dist_encoder', 'both')
        self.mlflow_online = kwargs.get('mlflow_online', False)
        self.joblib_dump = kwargs.get('joblib_dump', False)
        self.experiment_name = EXPERIMENT_NAME

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        distance_cols = ["pickup_latitude", "pickup_longitude",
                         'dropoff_latitude', 'dropoff_longitude']
        time_cols = ['pickup_datetime']

        dist_pipe = make_pipeline(DistanceTransformer(), StandardScaler())
        dist_center_pipe = make_pipeline(DistanceToCenter(), StandardScaler())
        time_pipe = make_pipeline(TimeFeaturesEncoder(),
                                  OneHotEncoder(handle_unknown='ignore'))

        # Removing either dist_pipe or dist_center_pipe depending on the
        # dist_encoder called
        feat_eng_blocks = [
            ('dist', dist_pipe, distance_cols),
            ('dist_to_center', dist_center_pipe, distance_cols),
            ('time', time_pipe, time_cols)
            ]
        if self.dist_encoder == 'dist':
            feat_eng_blocks.pop(1)
        elif self.dist_encoder == 'dist_to_center':
            feat_eng_blocks.pop(0)

        preproc_pipe = ColumnTransformer(feat_eng_blocks, remainder="drop")
        self.pipeline = make_pipeline(preproc_pipe, self.model)

    def run(self):
        """Sets and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse

    @memoized_property
    def mlflow_client(self, mlflow_online=False):
        if mlflow_online:
            mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def save_model(self, joblib_dump=False):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline, f'joblib/{str(self.model_name)}.joblib')


if __name__ == "__main__":
    df = get_data()
    df_clean = clean_df(df)
    X = df_clean.drop('fare_amount', axis=1)
    y = df_clean['fare_amount']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

    model_list = [LinearRegression(),
                  Lasso(alpha=0.05),
                  Ridge(alpha=0.05),
                  SVR(kernel='rbf'),
                  SVR(kernel='linear'),
                  RandomForestRegressor()
                  ]
    dist_encoder_list = ['dist', 'dist_to_center', 'both']

    for model in model_list:
        for dist_encoder in dist_encoder_list:
            params = dict(nrows=10000,
                          model=model,
                          dist_encoder=dist_encoder,
                          mlflow_online=False,
                          joblib_dump=False)
            trainer = Trainer(X_train, y_train, **params)
            trainer.run()
            rmse = trainer.evaluate(X_val, y_val)
            print(f'rmse: {rmse};',
                  f'dist_encoder: {dist_encoder};',
                  f'model: {trainer.model_name};')

            # logging parameters and metrics
            trainer.mlflow_log_param('model', trainer.model_name)
            trainer.mlflow_log_param('params', trainer.model_params)
            trainer.mlflow_log_param('dist_encoder', dist_encoder)
            trainer.mlflow_log_metric('rmse', rmse)

        # joblib
        # trainer.save_model()

    # getting location of mlflow
    if trainer.mlflow_online:
        experiment_id = trainer.mlflow_experiment_id
        print(f"experiment URL: https://mlflow.lewagon.ai/#/experiments/{experiment_id}")
