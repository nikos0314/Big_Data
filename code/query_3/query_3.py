
import time
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType, StringType
from pyspark.sql.functions import col, to_timestamp, broadcast


spark = (
    SparkSession.builder.appName("Crime Data Analysis Q3")
    .config("spark.sql.debug.maxToStringFields", "1000")
    .config("spark.executor.memory", "1g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


income_schema = "ZIP_CODE STRING, Community STRING, Estimated_Median_Income STRING"
income_df = spark.read.csv(
    "hdfs://master:9000/home/user/Big_Data/LA_income_2015.csv",
    header=True,
    schema=income_schema,
)


income_df = income_df.withColumn(
    "Estimated_Median_Income",
    F.regexp_replace(col("Estimated_Median_Income"), "[$,]", "").cast(IntegerType()),
)


geocoding_schema = "LAT DOUBLE, LON DOUBLE, ZIPcode STRING"
geocoding_df = spark.read.csv(
    "hdfs://master:9000/home/user/Big_Data/revgecoding.csv",
    header=True,
    schema=geocoding_schema,
)

geocoding_df = geocoding_df.withColumn(
    "zip_code", F.split(col("ZIPcode"), ";").getItem(0)
)

geocoding_df = geocoding_df.drop("ZIPcode").withColumnRenamed("zip_code", "ZIP_CODE")


crime_df = spark.read.csv(
    "hdfs://master:9000/home/user/crime_data",
    header=True,
    inferSchema=True,
).withColumn("DATE OCC", F.to_timestamp(F.col("DATE OCC"), "MM/dd/yyyy hh:mm:ss a"))

crime_2015_df = crime_df.filter(F.year(F.col("DATE OCC")) == 2015)


crime_2015_df = (
    crime_2015_df.filter(F.col("Vict Descent").isNotNull())
    .filter(col("Vict Descent") != "X")
    .select("AREA NAME", "LAT", "LON", "Vict Descent")
)




crime_2015_df = crime_2015_df.withColumn("LAT", col("LAT").cast("double"))
crime_2015_df = crime_2015_df.withColumn("LON", col("LON").cast("double"))

crime_with_zip_df = crime_2015_df.join(
    geocoding_df,
    (crime_2015_df.LAT == geocoding_df.LAT) & (crime_2015_df.LON == geocoding_df.LON),
    "left",
)



zip_codes = [
     row["ZIP_CODE"] for row in crime_with_zip_df.select("ZIP_CODE").distinct().collect()
]


top_3_income_zips = [
    row["ZIP_CODE"]
    for row in income_df.filter(col("ZIP_CODE").isin(zip_codes))
    .orderBy(col("Estimated_Median_Income").desc())
    .limit(3)
    .collect()
]


bottom_3_income_zips = [
    row["ZIP_CODE"]
    for row in income_df.filter(col("ZIP_CODE").isin(zip_codes))
    .orderBy(col("Estimated_Median_Income"))
    .limit(3)
    .collect()
]


top_3_income_crimes_df = crime_with_zip_df.filter(
    col("ZIP_CODE").isin(top_3_income_zips)
)
bottom_3_income_crimes_df = crime_with_zip_df.filter(
    col("ZIP_CODE").isin(bottom_3_income_zips)
)


top_3_income_victims = (
    top_3_income_crimes_df.groupBy("Vict Descent").count().orderBy(col("count").desc())
).withColumnRenamed("count", "total victims")

bottom_3_income_victims = (
    bottom_3_income_crimes_df.groupBy("Vict Descent")
    .count()
    .orderBy(col("count").desc())
).withColumnRenamed("count", "total victims")


def map_descent_code(code):
    descent_map = {
        "A": "Asian",
        "B": "Black",
        "C": "Chinese",
        "D": "Cambodian",
        "F": "Filipino",
        "G": "Guamanian",
        "H": "Hispanic/Latin/Mexican",
        "I": "American Indian/Alaskan Native",
        "J": "Japanese",
        "K": "Korean",
        "L": "Laotian",
        "O": "Other",
        "P": "Pacific Islander",
        "S": "Samoan",
        "U": "Hawaiian",
        "V": "Vietnamese",
        "W": "White",
        "X": "Unknown",
        "Z": "Asian Indian",
    }
    return descent_map.get(code, "Unknown")


map_descent_udf = F.udf(map_descent_code, StringType())

top_3_income_victims = top_3_income_victims.withColumn(
    "Vict Descent", map_descent_udf(col("Vict Descent"))
)
bottom_3_income_victims = bottom_3_income_victims.withColumn(
    "Vict Descent", map_descent_udf(col("Vict Descent"))
)


print("Top 3 Income ZIP Codes Victims by Descent")
top_3_income_victims.show(10, False)

print("Bottom 3 Income ZIP Codes Victims by Descent")
bottom_3_income_victims.show(10, False)


spark.stop()ecution Time: {:.2f} seconds".format(end_time_rdd - start_time_rdd))
spark.stop()








