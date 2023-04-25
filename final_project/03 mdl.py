# Databricks notebook source
# MAGIC %run ./includes/includes

# COMMAND ----------

df = spark.read.format("delta").load(HISTORIC_INVENTORY_INFO_DELTA_DIR)
display(df)

# COMMAND ----------

# from statsmodels.tsa.arima.model import ARIMA
# %pip install --upgrade pystan==2.19.1.1 fbprophet
from fbprophet import Prophet
import logging

# Suppresses `java_gateway` messages from Prophet as it runs.
logging.getLogger("py4j").setLevel(logging.ERROR)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

display(df)

# COMMAND ----------

df.toPandas()['hour_window']

# COMMAND ----------

from sklearn.model_selection import train_test_split

train_data = df.toPandas()
x_train, y_train, x_test, y_test = train_data[train_data['hour_window'] < '2023-02-20 00:00:00']['hour_window'], train_data[train_data['hour_window'] < '2023-02-20 00:00:00']['diff'], train_data[train_data['hour_window'] >= '2023-02-20 00:00:00']['hour_window'], train_data[train_data['hour_window'] >= '2023-02-20 00:00:00']['diff']

prophet_df = pd.DataFrame()
prophet_df["ds"] = pd.to_datetime(x_train)
prophet_df["y"] = y_train
# prophet_df = prophet_df[prophet_df['ds'] > '2023-01-01 00:00:00']
prophet_df.head()

# COMMAND ----------

import holidays

holiday = pd.DataFrame([])
for date, name in sorted(holidays.UnitedStates(years=[2021, 2022, 2023]).items()):
    holiday = holiday.append(pd.DataFrame({'ds': date, 'holiday': "US-Holidays"}, index=[0]), ignore_index=True)
holiday['ds'] = pd.to_datetime(holiday['ds'], format='%Y-%m-%d', errors='ignore')

holiday.head()

# COMMAND ----------

prophet_obj = Prophet(holidays=holiday)
prophet_obj.fit(prophet_df)
prophet_future = prophet_obj.make_future_dataframe(periods=217, freq="60min")
prophet_future.tail()

# COMMAND ----------

prophet_forecast = prophet_obj.predict(prophet_future)
prophet_forecast[['ds', 'yhat']].tail()

# COMMAND ----------

prophet_forecast[(prophet_forecast['ds'] > '2023-02-20 00:00:00') & (prophet_forecast['ds'] < '2023-02-21 00:00:00')]

# COMMAND ----------

yhat = prophet_forecast[prophet_forecast['ds'] >= '2023-02-20 00:00:00'][['yhat']]
yhat_round = np.round(yhat)
yhat_val = yhat_round['yhat']
yhat_val

# COMMAND ----------

from sklearn.metrics import mean_squared_error, r2_score
print(r2_score(y_test, yhat_val))
print(mean_squared_error(y_test, yhat_val))

# COMMAND ----------

pred_data = prophet_forecast[prophet_forecast['ds'] > '2023-01-01 00:00:00']
pred_data[['ds', 'yhat']]

# COMMAND ----------

prophet_plot = prophet_obj.plot(pred_data)

# COMMAND ----------

from statsmodels.tsa.arima.model import ARIMA

x_train = train_data[train_data['hour_window'] < '2023-02-20 00:00:00'][['hour_window', 'diff']]
x_train.index = x_train['hour_window']
x_train = x_train.drop('hour_window', axis=1)
model = ARIMA(np.array(x_train), order=(24, 2, 1))
fit = model.fit()
fit.forecast()

# COMMAND ----------

fit.summary()

# COMMAND ----------

x_train

# COMMAND ----------

real_time_status_data = spark.read.format("delta").load(REAL_TIME_STATION_STATUS_DELTA_DIR)
real_time_station_data = spark.read.format("delta").load(REAL_TIME_STATION_INFO_DELTA_DIR)
real_time_data = real_time_station_data.join(real_time_station_data, 'station_id')

# COMMAND ----------

display(real_time_station_data.filter(col("station_id") == "b30815c0-99b6-451b-be15-902992cb8abb").select("capacity").distinct())

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
