import time
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType, StringType

start_time_df = time.time()

spark = (
    SparkSession.builder.appName("Crime Data Analysis Q2 DF")
    .config("spark.sql.debug.maxToStringFields", "1000")
    .config("spark.executor.memory", "2g")
    .getOrCreate()
)

spark.conf.set("spark.sql.shuffle.partitions", "8")

spark.sparkContext.setLogLevel("WARN")


crime_df = (
    spark.read.csv(
        "hdfs://master:9000/home/user/crime_data", header=True, inferSchema=True
    )
    .select(F.col("TIME OCC").cast(IntegerType()), F.col("Premis Desc"))
    .filter((F.col("Premis Desc") == "STREET") & (F.col("TIME OCC").isNotNull()))
    .sortWithinPartitions("TIME OCC")
)


street_crimes_df = crime_df.withColumn(
    "time_segment",
    F.when((F.col("TIME OCC") >= 500) & (F.col("TIME OCC") < 1200), "Morning")
    .when((F.col("TIME OCC") >= 1200) & (F.col("TIME OCC") < 1700), "Afternoon")
    .when((F.col("TIME OCC") >= 1700) & (F.col("TIME OCC") < 2100), "Evening")
    .otherwise("Night")
)


result_df = street_crimes_df.groupBy("time_segment").count().orderBy(F.col("count").desc())


result_df.show()

end_time_df = time.time()
print("DataFrame API Execution Time: {:.2f} seconds".format(end_time_df - start_time_df))

spark.stop()
