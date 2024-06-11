import geopy
from pyspark.sql import SparkSession
from geopy.distance import geodesic
from pyspark.sql.types import DoubleType, IntegerType, StringType
import pyspark.sql.functions as F


spark = (
    SparkSession.builder.appName("LA Crime Analysis repartition join")
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


# Define function to calculate distance
def get_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km


def map_repartition(record):
    join_key = record["PREC"] if "PREC" in record else record["AREA"]
    tag = "stations" if "PREC" in record else "crimes"
    return (join_key, (tag, record))


crime_rdd_mapped = crime_rdd.map(map_repartition)
stations_rdd_mapped = stations_rdd.map(map_repartition)


union_rdd = crime_rdd_mapped.union(stations_rdd_mapped)


def partition_func(key):
    return hash(key)


partitioned_rdd = union_rdd.partitionBy(8, partition_func)


def reduce_repartition(key, records):
    crime_records = [record for tag, record in records if tag == "crimes"]
    station_records = [record for tag, record in records if tag == "stations"]

    results = []
    for crime_record in crime_records:
        for station_record in station_records:
            distance = get_distance(
                crime_record["LAT"],
                crime_record["LON"],
                station_record["y"],
                station_record["x"],
            )
            results.append((station_record["DIVISION"], (distance, 1)))
    return results


joined_rdd = partitioned_rdd.groupByKey().flatMap(
    lambda x: reduce_repartition(x[0], list(x[1]))
)


aggregated_rdd = (
    joined_rdd.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    .mapValues(lambda x: (x[0] / x[1], x[1]))
    .sortBy(lambda x: -x[1][1])
)


results_df = aggregated_rdd.map(lambda x: (x[0], round(x[1][0], 3), x[1][1])).toDF(
    ["DIVISION", "average_distance", "incidents"]
)


results_df.show(50)


spark.stop()
