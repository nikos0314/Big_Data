import geopy
from pyspark.sql import SparkSession
from geopy.distance import geodesic
from pyspark.sql.types import DoubleType, IntegerType, StringType
import pyspark.sql.functions as F


spark = (
    SparkSession.builder.appName("LA Crime Analysis using DataFrames")
    .config("spark.executor.memory", "1g")
    .config("spark.sql.debug.maxToStringFields", "1000")
    .getOrCreate()
)
spark.conf.set("spark.sql.shuffle.partitions", "4")

spark.sparkContext.setLogLevel("WARN")

stations_data_path = "hdfs://master:9000/home/user/Big_Data/LAPD_Police_Stations.csv"


stations_df = spark.read.csv(
    stations_data_path, header=True, inferSchema=True, sep=";"
).select("PREC", "DIVISION", "y", "x")


crime_df = spark.read.parquet(
    "hdfs://master:9000/home/user/crime_data_parquet",
    header=True,
).select("AREA ", "LAT", "LON", "Weapon Used Cd")


crime_df = crime_df.withColumnRenamed("AREA ", "AREA")
crime_df = crime_df.withColumn("AREA", F.col("AREA").cast(IntegerType()))
crime_df_filtered = crime_df.filter(
    (F.col("LAT") != 0)
    & (F.col("LON") != 0)
    & ~F.col("LAT").isNull()
    & ~F.col("LON").isNull()
)
crime_df_filtered = crime_df_filtered.filter(F.col("Weapon Used Cd").startswith("1"))



@F.udf(DoubleType())
def get_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km



joined_df = crime_df_filtered.join(
    stations_df, crime_df_filtered["AREA"] == stations_df["PREC"], "inner"
)


joined_df = joined_df.withColumn(
    "distance", get_distance(F.col("LAT"), F.col("LON"), F.col("y"), F.col("x"))
)


result_df = (
    joined_df.groupBy("DIVISION")
    .agg(
        F.round(F.avg("distance"), 3).alias("average_distance"),
        F.count("*").alias("incidents"),
    )
    .orderBy(F.col("incidents").desc())
)


result_df.show(50)


spark.stop()
