from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from taxifare.data import clean_df, get_data
from taxifare.trainer import Trainer


params = dict(nrows=10000,
                model=RandomForestRegressor(),
                dist_encoder='both',
                mlflow_online=False,
                joblib_dump=True)


if __name__ == "__main__":
    df = get_data(**params)
    df = clean_df(df)
    X = df.drop("fare_amount", axis=1)
    y = df["fare_amount"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    trainer = Trainer(X_train, y_train, **params)
    trainer.run()
    rmse = trainer.evaluate(X_val, y_val)
    trainer.save_model()
    print(f'rmse: {rmse};',
            f'dist_encoder: {trainer.dist_encoder};',
            f'model: {trainer.model_name};')
