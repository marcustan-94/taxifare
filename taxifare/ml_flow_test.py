import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property

EXPERIMENT_NAME = '[SG] [Singapore] [marcustan-94] taxifare v0'
yourname = 'marcus'

############################## Run ML Flow locally ###########################
# client = MlflowClient()

# try:
#     experiment_id = client.create_experiment(EXPERIMENT_NAME)
# except BaseException:
#     experiment_id = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

# if yourname is None:
#     print("please define your name, it'll will be used as a parameter to log")

# for model in ["linear", "randomforest"]:
#     run = client.create_run(experiment_id)
#     client.log_metric(run.info.run_id, "rmse", 5.0)
#     client.log_param(run.info.run_id, "model", model)
#     client.log_param(run.info.run_id, "student_name", yourname)


################ Indicate mlflow to log to remote server ##################
################    Wrapping code in a class Trainer     ##################

MLFLOW_URI = "https://mlflow.lewagon.ai/"

class Trainer():

    def __init__(self, experiment_name):
        self.experiment_name = experiment_name

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

for model in ["linear", "randomforest"]:
    trainer = Trainer(EXPERIMENT_NAME)
    trainer.mlflow_log_metric("rmse", 5.0)
    trainer.mlflow_log_param("model", model)
    trainer.mlflow_log_param("student_name", yourname)

print(f'https://mlflow.lewagon.ai/#/experiments/{trainer.mlflow_experiment_id}')
