# Databricks notebook source
# MAGIC %run ./includes/includes

# COMMAND ----------

# Import Statements

from datetime import datetime, timedelta
# from fbprophet import Prophet
import logging
import holidays
import mlflow

# Prophet Forecasting
from prophet import Prophet, serialize
from prophet.diagnostics import cross_validation, performance_metrics

# Visualization
import plotly.express as px

# Hyperparameter tuning
import itertools

# Performance metrics
from sklearn.metrics import mean_absolute_error

# MLFlow Tracking
from mlflow.tracking.client import MlflowClient

# COMMAND ----------

# Fetch Input arguments

# dbutils.widgets.removeAll()
# dbutils.widgets.dropdown("Promote Model", "No", ["No", "Yes"])
# dbutils.widgets.text("Hours to forecast", "8")

# start_date = str(dbutils.widgets.get('01.start_date'))
# end_date = str(dbutils.widgets.get('02.end_date'))
hours_to_forecast = int(dbutils.widgets.get('Hours to forecast'))
promote_model = bool(True if str(dbutils.widgets.get('Promote Model')).lower() == 'yes' else False)

print(hours_to_forecast, promote_model)

# COMMAND ----------

# Constants

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
PERIOD_TO_FORECAST_FOR = 168 + hours_to_forecast
ARTIFACT_PATH = GROUP_MODEL_NAME
STAGING = 'Staging'
PROD = 'Production'
ARCHIVE = 'Archived'
np.random.seed(265)

## Helper routine to extract the parameters that were used to train a specific instance of the model
def extract_params(pr_model):
    return {attr: getattr(pr_model, attr) for attr in serialize.SIMPLE_ATTRIBUTES}

# COMMAND ----------

# Read datasets

inventory_info = spark.read.format("delta").load(INVENTORY_INFO_DELTA_DIR).select(col("hour_window").alias("ds"), col("diff").alias("y"))
weather_info = spark.read.format("delta").load(WEATHER_INFO_DELTA_DIR).select(col("hour_window").alias("ds"), "feels_like", "clouds", "is_weekend")
merged_info = inventory_info.join(weather_info, on="ds", how="inner")

try:
    model_info = spark.read.format("delta").load(MODEL_INFO)
except:
    model_info = None

# COMMAND ----------

# Get Split time based upon period to forecast

latest_end_timestamp_in_silver_storage = inventory_info.select("ds").sort(desc("ds")).head(1)[0][0]
time_for_split = (datetime.strptime(latest_end_timestamp_in_silver_storage, TIME_FORMAT) - timedelta(hours=PERIOD_TO_FORECAST_FOR)).strftime(TIME_FORMAT)

merged_info = merged_info.filter(col("ds") <= latest_end_timestamp_in_silver_storage).dropna()

# COMMAND ----------

# Create train-test data

train_data = merged_info.filter(col("ds") <= time_for_split).toPandas()
test_data = merged_info.filter(col("ds") > time_for_split).toPandas()
x_train, y_train, x_test, y_test = train_data["ds"], train_data["y"], test_data["ds"], test_data["y"]

# COMMAND ----------

# Suppresses `java_gateway` messages from Prophet as it runs.

logging.getLogger("py4j").setLevel(logging.ERROR)

# COMMAND ----------

fig = px.line(train_data, x="ds", y="y", title='Bike Rides')
fig.show()

# COMMAND ----------

#--------------------------------------------#
# Automatic Hyperparameter Tuning
#--------------------------------------------#

# Set up parameter grid
param_grid = {  
    'changepoint_prior_scale': [0.0005],
    'seasonality_prior_scale': [0.8],
    'seasonality_mode': ['multiplicative'],
    'yearly_seasonality' : [False],
    'weekly_seasonality': [True],
    'daily_seasonality': [False]
}

# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

print(f"Total training runs {len(all_params)}")

# Create a list to store MAPE values for each combination
maes = [] 

# Use cross validation to evaluate all parameters
for params in all_params:
    with mlflow.start_run(): 
        # Fit a model using one parameter combination + holidays
        m = Prophet(**params) 
        holidays = pd.DataFrame({"ds": [], "holiday": []})
        m.add_country_holidays(country_name='US')
        #m.add_regressor('feels_like')
        #m.add_regressor('clouds')
        #m.add_regressor('is_weekend')
        m.fit(train_data)

        # Cross-validation
        # df_cv = cross_validation(model=m, initial='710 days', period='180 days', horizon = '365 days', parallel="threads")
        # Model performance
        # df_p = performance_metrics(m, rolling_window=1)

        # try:
        #     metric_keys = ["mse", "rmse", "mae", "mape", "mdape", "smape", "coverage"]
        #     metrics = {k: df_p[k].mean() for k in metric_keys}
        #     params = extract_params(m)
        # except:
        #     pass

        # print(f"Logged Metrics: \n{json.dumps(metrics, indent=2)}")
        # print(f"Logged Params: \n{json.dumps(params, indent=2)}")

        y_pred = m.predict(test_data)

        mae = mean_absolute_error(y_test, y_pred['yhat'])

        mlflow.prophet.log_model(m, artifact_path=ARTIFACT_PATH)
        mlflow.log_params(params)
        mlflow.log_metrics({'mae': mae})
        model_uri = mlflow.get_artifact_uri(ARTIFACT_PATH)
        print(f"Model artifact logged to: {model_uri}")

        # Save model performance metrics for this combination of hyper parameters
        maes.append((mae, model_uri))
        

# COMMAND ----------

# Tuning results

tuning_results = pd.DataFrame(all_params)
tuning_results['mae'] = list(zip(*maes))[0]
tuning_results['model']= list(zip(*maes))[1]

best_params = dict(tuning_results.iloc[tuning_results[['mae']].idxmin().values[0]])

best_params

# COMMAND ----------

# Create Forecast

loaded_model = mlflow.prophet.load_model(best_params['model'])

forecast = loaded_model.predict(test_data)

print(f"forecast:\n${forecast.tail(40)}")

# COMMAND ----------

# Plot forecast

prophet_plot = loaded_model.plot(forecast)

# COMMAND ----------

# Plot each components of the forecast separately

prophet_plot2 = loaded_model.plot_components(forecast)

# COMMAND ----------

# Finding residuals

results = forecast[['ds','yhat']].join(train_data, lsuffix = '_caller', rsuffix = '_other')
results['residual'] = results['yhat'] - results['y']

# COMMAND ----------

# Plot the residuals

fig = px.scatter(
    results, x='yhat', y='residual',
    marginal_y='violin',
    trendline='ols'
)
fig.show()

# COMMAND ----------

# Register Model to MLFlow

model_details = mlflow.register_model(model_uri=best_params['model'], name=ARTIFACT_PATH)

# COMMAND ----------

# Call MLFlow Client

client = MlflowClient()

# COMMAND ----------

# Apply appropriate tag

try:
    latest_staging_mae = model_info.filter(col("tag") == STAGING).select("mae").head(1)[0][0]
except:
    latest_staging_mae = 0

cur_version = None
if promote_model:
    stage = PROD
    cur_version = client.get_latest_versions(ARTIFACT_PATH, stages=[PROD])
elif best_params['mae'] > latest_staging_mae:
    stage = STAGING
    cur_version = client.get_latest_versions(ARTIFACT_PATH, stages=[STAGING])
else:
    stage = ARCHIVE

if cur_version:
    client.transition_model_version_stage(
        name=GROUP_MODEL_NAME,
        version=cur_version[0].version,
        stage=ARCHIVE,
        )

client.transition_model_version_stage(
  name=model_details.name,
  version=model_details.version,
  stage=stage,
)

# COMMAND ----------

# Current Model Stage

model_version_details = client.get_model_version(

  name=model_details.name,

  version=model_details.version,

)

print("The current model stage is: '{stage}'".format(stage=model_version_details.current_stage))

# COMMAND ----------

# Latest Model Version

latest_version_info = client.get_latest_versions(ARTIFACT_PATH, stages=["Staging"])

latest_staging_version = latest_version_info[0].version

print("The latest staging version of the model '%s' is '%s'." % (ARTIFACT_PATH, latest_staging_version))

# COMMAND ----------

model_staging_uri = "models:/{model_name}/staging".format(model_name=ARTIFACT_PATH)

print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_staging_uri))

model_staging = mlflow.prophet.load_model(model_staging_uri)

# COMMAND ----------

# Update Gold Table

def get_forecast_df(results, tag, mae):
    df = results.copy()
    df['tag'] = tag
    df['mae'] = mae
    return df[['ds_caller', 'y', 'yhat', 'tag', 'residual', 'mae']]

try:
    staging_data = model_info.filter(col("tag") == STAGING)
    prod_data = model_info.filter(col("tag") == PROD)
except:
    staging_data = None
    prod_data = None

forecast_df = pd.DataFrame(columns=['ds', 'y', 'yhat', 'tag', 'residual', 'mae'])

final_df = None
if promote_model:
    forecast_df[['ds', 'y', 'yhat', 'tag', 'residual', 'mae']] = get_forecast_df(results, PROD, best_params['mae'])
    final_df = staging_data.union(spark.createDataFrame(forecast_df)) if staging_data else spark.createDataFrame(forecast_df)
elif best_params['mae'] > latest_staging_mae:
    forecast_df[['ds', 'y', 'yhat', 'tag', 'residual', 'mae']] = get_forecast_df(results, STAGING, best_params['mae'])
    final_df = prod_data.union(spark.createDataFrame(forecast_df)) if prod_data else spark.createDataFrame(forecast_df)
else:
    pass

if final_df:
    final_df\
        .write\
        .format("delta")\
        .option("path", MODEL_INFO)\
        .mode("overwrite")\
        .save()


# COMMAND ----------

# results['yhat_avail'] = np.round(results['yhat']) + 61
# results

# COMMAND ----------

# fig = px.line(x=results['ds_caller'], y=results['yhat_avail'])
# fig.add_hline(y=61)
# fig.show()

# COMMAND ----------

# from statsmodels.tsa.arima.model import ARIMA

# x_train = train_data[train_data['hour_window'] < '2023-02-20 00:00:00'][['hour_window', 'diff']]
# x_train.index = x_train['hour_window']
# x_train = x_train.drop('hour_window', axis=1)
# model = ARIMA(np.array(x_train), order=(24, 2, 1))
# fit = model.fit()
# fit.forecast()

# COMMAND ----------

import json

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
