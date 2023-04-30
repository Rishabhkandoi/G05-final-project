# Databricks notebook source
# MAGIC %md
# MAGIC ##DSCC 202 - 402 Final Project Spring 2023
# MAGIC <p>
# MAGIC <img src='https://data-science-at-scale.s3.amazonaws.com/images/fp2023.png'>
# MAGIC </p>
# MAGIC see product description and rubric in repo same directory as this notebook.

# COMMAND ----------

from datetime import datetime as dt
from datetime import timedelta
import json

# dbutils.widgets.removeAll()

# # dbutils.widgets.text('01.start_date', "2021-10-01")
# # dbutils.widgets.text('02.end_date', "2023-03-01")
dbutils.widgets.dropdown('Hours to Forecast', '48', ['4', '8', '12', '24', '48'])
dbutils.widgets.dropdown('Promote Model', 'No', ['No', 'Yes'])

# start_date = str(dbutils.widgets.get('01.start_date'))
# end_date = str(dbutils.widgets.get('02.end_date'))
hours_to_forecast = int(dbutils.widgets.get('Hours to Forecast'))
promote_model = str(dbutils.widgets.get('Promote Model'))

print(hours_to_forecast, promote_model)

# COMMAND ----------

# DBTITLE 1,Run the ETL Notebook
# Run the Data Prepartion note the one hour 3600 second timeout!
result = dbutils.notebook.run("01 etl", 3600, {"Hours to Forecast":hours_to_forecast,"Promote Model":promote_model})

# Check the results
assert json.loads(result)["exit_code"] == "OK", "Data Preparation Failed!" # Check to see that it worked

# COMMAND ----------

# DBTITLE 1,Run the EDA Notebook
# Run the Data Prepartion
result = dbutils.notebook.run("02 eda", 3600, {"Hours to Forecast":hours_to_forecast,"Promote Model":promote_model})

# Check the results
assert json.loads(result)["exit_code"] == "OK", "Data Preparation Failed!" # Check to see that it worked

# COMMAND ----------

# DBTITLE 1,Run Model Development Notebook
# Run the Data Prepartion
result = dbutils.notebook.run("03 mdl", 3600, {"Hours to Forecast":hours_to_forecast,"Promote Model":promote_model})

# Check the results
assert json.loads(result)["exit_code"] == "OK", "Data Preparation Failed!" # Check to see that it worked

# COMMAND ----------

# DBTITLE 1,Run Station Inventory Forecast Notebook
# Run the Data Prepartion
result = dbutils.notebook.run("04 app", 3600, {"Hours to Forecast":hours_to_forecast,"Promote Model":promote_model})

# Check the results
assert json.loads(result)["exit_code"] == "OK", "Data Preparation Failed!" # Check to see that it worked
