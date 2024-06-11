import geopy
from pyspark.sql import SparkSession
from geopy.distance import geodesic
from pyspark.sql.types import DoubleType, IntegerType, StringType
import pyspark.sql.functions as F

# Initialize Spark Session
spark = (
    SparkSession.builder.appName("LA Crime Analysis broadcast join")
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


crime_rdd = crime_df_filtered.rdd
stations_rdd = stations_df.rdd


def get_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km


stations_rdd_keyed = stations_rdd.map(lambda row: (row["PREC"], row.asDict()))
stations_broadcast = spark.sparkContext.broadcast(stations_rdd_keyed.collectAsMap())


def broadcast_join(crime_record, stations_dict):
    area = crime_record["AREA"]
    if area in stations_dict:
        station = stations_dict[area]
        distance = get_distance(
            crime_record["LAT"], crime_record["LON"], station["y"], station["x"]
        )
        return (station["DIVISION"], (distance, 1))
    return None


broadcast_joined_rdd = crime_rdd.map(
    lambda x: broadcast_join(x.asDict(), stations_broadcast.value)
).filter(lambda x: x is not None)


broadcast_joined_rdd = broadcast_joined_rdd.persist()


broadcast_results_rdd = (
    broadcast_joined_rdd.mapValues(lambda x: (x[0], x[1]))
    .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    .mapValues(lambda x: (x[0] / x[1], x[1]))
    .sortBy(lambda x: x[1][1], ascending=False)
)


broadcast_results_df = broadcast_results_rdd.map(
    lambda x: (x[0], x[1][0], x[1][1])
).toDF(["DIVISION", "average_distance", "incidents"])


broadcast_results_df = broadcast_results_df.withColumn(
    "average_distance", F.round(F.col("average_distance"), 3)
)


broadcast_results_df.show(50)


spark.stop()
