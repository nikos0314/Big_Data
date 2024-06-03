import time
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType, TimestampType, StringType

start_time_df = time.time()

spark = (
    SparkSession.builder.appName("Crime Data Analysis Q2 DF")
    .config("spark.sql.debug.maxToStringFields", "1000")
    .config("spark.executor.memory", "1g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


crimes_df = spark.read.csv(
    "hdfs://master:9000/home/user/crime_data", header=True, inferSchema=True
)


crime_df = crimes_df.select(F.col("TIME OCC").cast(IntegerType()), F.col("Premis Desc"))

crime_df = crime_df.filter(
    (F.col("Premis Desc") == "STREET") & (F.col("TIME OCC").isNotNull())
)


def get_time_segment(time_occ):
    if time_occ is None:
        return None
    hour = time_occ
    if 500 <= hour < 1200:
        return "Morning"
    elif 1200 <= hour < 1700:
        return "Afternoon"
    elif 1700 <= hour < 2100:
        return "Evening"
    else:
        return "Night"


time_segment_udf = F.udf(get_time_segment, StringType())

street_crimes_df = crime_df.withColumn(
    "time_segment", time_segment_udf(F.col("TIME OCC"))
)


result_df = (
    street_crimes_df.groupBy("time_segment").count().orderBy(F.col("count").desc())
)

result_df.show()

end_time_df = time.time()
print(
    "DataFrame API Execution Time: {:.2f} seconds".format(end_time_df - start_time_df)
)
spark.stop()
