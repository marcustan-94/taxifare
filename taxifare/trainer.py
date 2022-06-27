# imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer, DistanceToCenter
from TaxiFareModel.utils import compute_rmse
from memoized_property import memoized_property
import joblib
import mlflow
from  mlflow.tracking import MlflowClient

MLFLOW_URI = "https://mlflow.lewagon.ai/"
EXPERIMENT_NAME = '[SG] [Singapore] [marcustan-94] taxifare v0'

class Trainer():
    def __init__(self, X, y, model, dist_encoder, experiment_name=EXPERIMENT_NAME):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.model = model
        self.model_name = str(model)[:str(model).find('(')]
        self.model_params = str(model)[str(model).find('('):]
        self.dist_encoder = dist_encoder
        self.experiment_name = experiment_name

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
            ('dist_encoder', self.dist_encoder),
            ('stdscaler', StandardScaler())
            ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
            ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
            ], remainder="drop")
        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('model', self.model)
            ])

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)
        return rmse

    @memoized_property
    def mlflow_client(self):
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

    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline, f'joblib/{str(self.model)}.joblib')


if __name__ == "__main__":
    # get data
    df = get_data()

    # clean data
    df_clean = clean_data(df)

    # set X and y
    X = df_clean.drop('fare_amount', axis=1)
    y = df_clean['fare_amount']

    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

    model_list = [LinearRegression(),
                  Lasso(alpha=0.05),
                  Ridge(alpha=0.05),
                  SVR(kernel='rbf'),
                  SVR(kernel='linear')
                  ]
    dist_encoders = [DistanceToCenter(), DistanceTransformer()]

    for model in model_list:

        for dist_encoder in dist_encoders:

            # instanciating Trainer class
            trainer = Trainer(X_train, y_train, dist_encoder=dist_encoder, model=model)

            # train
            trainer.run()

            # evaluate
            rmse = trainer.evaluate(X_val, y_val)

            # logging parameters and metrics
            trainer.mlflow_log_param('model', trainer.model_name)
            trainer.mlflow_log_param('params', trainer.model_params)
            trainer.mlflow_log_param('dist_metric', trainer.dist_encoder)
            trainer.mlflow_log_metric('rmse', rmse)

        # joblib
        # trainer.save_model()

    # getting location of mlflow
    experiment_id = trainer.mlflow_experiment_id
    print(f"experiment URL: https://mlflow.lewagon.ai/#/experiments/{experiment_id}")
