import time
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType, TimestampType, StringType

start_time_rdd = time.time()

spark = (
    SparkSession.builder.appName("Crime Data Analysis Q2 RDD")
    .config("spark.sql.debug.maxToStringFields", "1000")
    .config("spark.executor.memory", "1g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


crimes_df = spark.read.csv(
    "hdfs://master:9000/home/user/crime_data", header=True, inferSchema=True
)

crime_df = crimes_df.select(F.col("TIME OCC").cast(IntegerType()), F.col("Premis Desc"))

street_crimes_rdd = crime_df.filter(
    (F.col("Premis Desc") == "STREET") & (F.col("TIME OCC").isNotNull())
).rdd


def map_time_segment(row):
    time_occ = row["TIME OCC"]
    if time_occ is None:
        return None
    hour = time_occ
    if 500 <= hour < 1200:
        return ("Morning", 1)
    elif 1200 <= hour < 1700:
        return ("Afternoon", 1)
    elif 1700 <= hour < 2100:
        return ("Evening", 1)
    else:
        return ("Night", 1)


time_segments_rdd = (
    street_crimes_rdd.map(map_time_segment)
    .reduceByKey(lambda a, b: a + b)
    .sortBy(lambda x: -x[1])
)

results = time_segments_rdd.collect()
for result in results:
    print(result)

end_time_rdd = time.time()
print("RDD API Execution Time: {:.2f} seconds".format(end_time_rdd - start_time_rdd))
spark.stop()
