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

# Constants

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
PERIOD_TO_FORECAST_FOR = 172
ARTIFACT_PATH = GROUP_MODEL_NAME
np.random.seed(265)

## Helper routine to extract the parameters that were used to train a specific instance of the model
def extract_params(pr_model):
    return {attr: getattr(pr_model, attr) for attr in serialize.SIMPLE_ATTRIBUTES}

# COMMAND ----------

# Read datasets

inventory_info = spark.read.format("delta").load(INVENTORY_INFO_DELTA_DIR).select(col("hour_window").alias("ds"), col("diff").alias("y"))
weather_info = spark.read.format("delta").load(WEATHER_INFO_DELTA_DIR).select(col("hour_window").alias("ds"), "feels_like", "clouds", "is_weekend")
merged_info = inventory_info.join(weather_info, on="ds", how="inner")

# COMMAND ----------

display(merged_info.orderBy("ds", ascending=True))

# COMMAND ----------

display(merged_info.orderBy("ds", ascending=False))

# COMMAND ----------

merged_info.rdd.getNumPartitions()

# COMMAND ----------

print(spark.sparkContext.defaultParallelism)

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

# def fill_nulls(df, column):
#     w = Window.orderBy("ds")
#     return df.withColumn(column, lag(column, 1).over(w))

# # Iterate over all columns except the first one and fill null values
# for column in merged_info.columns[1:]:
#     joined_df = fill_nulls(merged_info, column)

# joined_df.toPandas().info()

# COMMAND ----------

# Suppresses `java_gateway` messages from Prophet as it runs.

logging.getLogger("py4j").setLevel(logging.ERROR)

# COMMAND ----------

# holiday = pd.DataFrame([])
# for date, name in sorted(holidays.UnitedStates(years=[2021, 2022, 2023]).items()):
#     holiday = holiday.append(pd.DataFrame({'ds': date, 'holiday': "US-Holidays"}, index=[0]), ignore_index=True)
# holiday['ds'] = pd.to_datetime(holiday['ds'], format='%Y-%m-%d', errors='ignore')

# holiday.head()

# COMMAND ----------

# prophet_obj = Prophet(holidays=holiday)
# prophet_obj.fit(prophet_df)
# prophet_future = prophet_obj.make_future_dataframe(periods=PERIOD_TO_FORECAST_FOR, freq="60min")
# prophet_future.tail()

# COMMAND ----------

# prophet_forecast = prophet_obj.predict(prophet_future)
# prophet_forecast[['ds', 'yhat']].tail()

# COMMAND ----------

fig = px.line(train_data, x="ds", y="y", title='Bike Rides')
fig.show()

# COMMAND ----------

# # Initiate the model
# baseline_model = Prophet()

# # Fit the model on the training dataset
# baseline_model.fit(train_data)

# # Cross validation
# baseline_model_cv = cross_validation(model=baseline_model, initial='510 days', horizon=PERIOD_TO_FORECAST_FOR, parallel="threads")
# baseline_model_cv.head()

# # Model performance metrics
# baseline_model_p = performance_metrics(baseline_model_cv, rolling_window=1)
# baseline_model_p.head()

# # Get the performance value
# print(f"MAPE of baseline model: {baseline_model_p['mape'].values[0]}")

# COMMAND ----------

#--------------------------------------------#
# Automatic Hyperparameter Tuning
#--------------------------------------------#

# Set up parameter grid
param_grid = {  
    'changepoint_prior_scale': [ 0.0005, 0.001],
    'seasonality_prior_scale': [1, 10],
    'seasonality_mode': ['additive'],
    'yearly_seasonality' : [False],
    'weekly_seasonality': [True, False],
    'daily_seasonality': [True, False]
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

loaded_model = mlflow.prophet.load_model(best_params['model'])

forecast = loaded_model.predict(test_data)

print(f"forecast:\n${forecast.tail(40)}")

# COMMAND ----------

prophet_plot = loaded_model.plot(forecast)

# COMMAND ----------

prophet_plot2 = loaded_model.plot_components(forecast)

# COMMAND ----------

results=forecast[['ds','yhat']].join(train_data, lsuffix='_caller', rsuffix='_other')
results['residual'] = results['yhat'] - results['y']

# COMMAND ----------

#plot the residuals
fig = px.scatter(
    results, x='yhat', y='residual',
    marginal_y='violin',
    trendline='ols'
)
fig.show()

# COMMAND ----------

model_details = mlflow.register_model(model_uri=best_params['model'], name=ARTIFACT_PATH)

# COMMAND ----------

client = MlflowClient()

# COMMAND ----------

client.transition_model_version_stage(

  name=model_details.name,

  version=model_details.version,

  stage='Staging',

)

# COMMAND ----------

model_version_details = client.get_model_version(

  name=model_details.name,

  version=model_details.version,

)

print("The current model stage is: '{stage}'".format(stage=model_version_details.current_stage))

# COMMAND ----------

latest_version_info = client.get_latest_versions(ARTIFACT_PATH, stages=["Staging"])

latest_staging_version = latest_version_info[0].version

print("The latest staging version of the model '%s' is '%s'." % (ARTIFACT_PATH, latest_staging_version))

# COMMAND ----------

model_staging_uri = "models:/{model_name}/staging".format(model_name=ARTIFACT_PATH)

print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_staging_uri))

model_staging = mlflow.prophet.load_model(model_staging_uri)

# COMMAND ----------

latest_version = int(client.get_latest_versions(ARTIFACT_PATH)[0].version)

df_list = []
for i in range(1, latest_version+1):
    model_url = client.get_model_version(version=i, name=ARTIFACT_PATH).source
    loaded_model = mlflow.prophet.load_model(model_url)
    forecast = loaded_model.predict(test_data)
    results=forecast[['ds','yhat']].join(train_data, lsuffix='_caller', rsuffix='_other')
    results['residual'] = results['yhat'] - results['y']
    df = pd.DataFrame({'name': ['v'+str(i)]*len(results),
                    'yhat': results['yhat'],
                    'residual': results['residual']})
    df_list.append(df)

# model_url_v1 = client.get_model_version(version=1, name=ARTIFACT_PATH).source
# model_url_v2 = client.get_model_version(version=2, name=ARTIFACT_PATH).source
# model_url_v3 = client.get_model_version(version=3, name=ARTIFACT_PATH).source

# loaded_model_v1 = mlflow.prophet.load_model(model_url_v1)
# forecast_v1 = loaded_model_v1.predict(test_data)
# results_v1=forecast_v1[['ds','yhat']].join(train_data, lsuffix='_caller', rsuffix='_other')
# results_v1['residual'] = results_v1['yhat'] - results_v1['y']

# loaded_model_v2 = mlflow.prophet.load_model(model_url_v2)
# forecast_v2 = loaded_model_v2.predict(test_data)
# results_v2=forecast_v2[['ds','yhat']].join(train_data, lsuffix='_caller', rsuffix='_other')
# results_v2['residual'] = results_v2['yhat'] - results_v2['y']

# loaded_model_v3 = mlflow.prophet.load_model(model_url_v3)
# forecast_v3 = loaded_model_v3.predict(test_data)
# results_v3=forecast_v3[['ds','yhat']].join(train_data, lsuffix='_caller', rsuffix='_other')
# results_v3['residual'] = results_v3['yhat'] - results_v3['y']

# df1 = pd.DataFrame({'name': ['v1']*len(results_v1),
#                     'yhat': results_v1['yhat'],
#                     'residual': results_v1['residual']})

# df2 = pd.DataFrame({'name': ['v2']*len(results_v2),
#                     'yhat': results_v2['yhat'],
#                     'residual': results_v2['residual']})

# df3 = pd.DataFrame({'name': ['v3']*len(results_v3),
#                     'yhat': results_v3['yhat'],
#                     'residual': results_v3['residual']})

# df = pd.concat([df1, df2, df3])
df = pd.concat(df_list)

#plot the residuals
fig = px.scatter(
    df, x='yhat', y='residual',
    marginal_y='violin',
    trendline='ols',
    color='name'
)
fig.show()

# COMMAND ----------

results['yhat_avail'] = np.round(results['yhat']) + 61
results

# COMMAND ----------

fig = px.line(x=results['ds_caller'], y=results['yhat_avail'])
fig.add_hline(y=61)
fig.show()

# COMMAND ----------

# from statsmodels.tsa.arima.model import ARIMA

# x_train = train_data[train_data['hour_window'] < '2023-02-20 00:00:00'][['hour_window', 'diff']]
# x_train.index = x_train['hour_window']
# x_train = x_train.drop('hour_window', axis=1)
# model = ARIMA(np.array(x_train), order=(24, 2, 1))
# fit = model.fit()
# fit.forecast()

# COMMAND ----------

# start_date = str(dbutils.widgets.get('01.start_date'))
# end_date = str(dbutils.widgets.get('02.end_date'))
# hours_to_forecast = int(dbutils.widgets.get('03.hours_to_forecast'))
# promote_model = bool(True if str(dbutils.widgets.get('04.promote_model')).lower() == 'yes' else False)

# print(start_date,end_date,hours_to_forecast, promote_model)
# print("YOUR CODE HERE...")

# COMMAND ----------

# import json

# # Return Success
# dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
