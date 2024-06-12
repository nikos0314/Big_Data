from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp
import pyspark.sql.functions as F

spark = (
    SparkSession.builder.appName("Crime Data Analysis DF parquet")
    .config("spark.executor.memory", "1g")
    .config("spark.sql.debug.maxToStringFields", "1000")
    .getOrCreate()
)


spark.sparkContext.setLogLevel("WARN")


path = "hdfs://master:9000/home/user/crime_data/Crime_Data_from_2010_to_2019.csv"


crime_df = spark.read.csv(path, header=True, inferSchema=True)

crime_df = crime_df.withColumn(
    "DATE OCC", to_timestamp(col("DATE OCC"), "MM/dd/yyyy hh:mm:ss a")
).filter(F.year(F.col("DATE OCC")) == 2015)


output_parquet_path = "hdfs://master:9000/home/user/crime_data_parquet_2015"


crime_df.coalesce(1).write.mode("overwrite").parquet(output_parquet_path)


spark.stop()
