import sys, os, yaml
import urllib.request
import pandas as pd 

from configs.project import get_project_config, get_project_root_path


def get_df_from_config(config):
  from databricks.connect import DatabricksSession as SparkSession
  spark = SparkSession.builder.getOrCreate()
  try:
    _ = urllib.request.urlretrieve(config.url, config.local_path)
    print(f"Downloaded {config.url} to {config.local_path}")
    df = spark.read.parquet(config.local_path)
    print(f"Read DataFrame from {config.local_path}")
  except Exception as e:
    print(f"Error downloading {config.url} to {config.local_path}: {e}")
    try:
      from databricks.sdk.runtime import dbutils
      project_root_path = get_project_root_path()
      data_path = os.path.join(project_root_path, config.local_clone_path)
      dbutils.fs.cp(f"file:{data_path}", config.local_path)
      df = spark.read.parquet(config.local_path)
      print(f"Read DataFrame from {config.local_path}")
    except Exception as e:
      print(f"Error coping {data_path} to {config.local_path}: {e}")
      try:
        pddf = pd.read_parquet(data_path)
        print(f"Read pandas DataFrame from {data_path}")
        df = spark.createDataFrame(pddf)
        print("Created DataFrame from pandas DataFrame.")
      except Exception as e:
        print(f"Error loading parquet file: {e}")
  return df  