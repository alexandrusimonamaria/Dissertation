!pip install syft==0.8.2
!pip install numpy pandas scikit-learn pySyft

import syft as sy
import torch

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType

#Step 1: Load the dataset
def load_data(url):
        column_names = [
            "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",
            "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
            "Hours per week", "Native Country", "Income"
        ]
        df = pd.read_csv(url, names=column_names, sep=",\s*", engine="python", na_values=' ?')
        return df

#Step 2: Preprocess the dataset
def preprocess_data(df):
    le = LabelEncoder()
    df["Income"] = le.fit_transform(df["Income"])
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Age"].fillna(df["Age"].mean(), inplace=True)  # Missing value are filled with media
    return df

#Step 3: SMPC initialization
def setup_smpc():
    alice = sy.Domain(name="alice")
    bob = sy.Domain(name="bob")
    client_alice = alice.get_guest_client()
    client_bob = bob.get_guest_client()
    return client_alice, client_bob

#Step 4: Secure avg calculation
def secure_average(income_tensor):
    half = len(income_tensor) // 2
    split1 = income_tensor[:half]
    split2 = income_tensor[half:]
    avg1 = split1.mean()
    avg2 = split2.mean()
    overall_avg = (avg1 + avg2) / 2
    print(f" Simulated Secure Average Income Label: {overall_avg.item():.4f}")
    return overall_avg

#Step 5: K-anonymity
def apply_k_anonymity(data_df, quasi_identifiers, k_value):
    print(f" K-Anonymity with k={k_value} on: {quasi_identifiers}")
    processed_data = data_df.withColumn(
        "age_generalized",
         F.floor(F.col("age")/20)*20
    )
    # Group by quasi-identifiers and count group sizes
    k_anonymous_data = processed_data.groupBy(
        [qi if qi != "age" else "age_generalized" for qi in quasi_identifiers]
    ).agg(F.count("*").alias("group_size")) \
    .filter(F.col("group_size") >= k_value) \
    .join(processed_data, on=[qi if qi != "age" else "age_generalized" for qi in quasi_identifiers], how="inner") \
    .drop("group_size", "age") \
    .withColumnRenamed("age_generalized", "age")
    print(f" K-anonimity row: {k_anonymous_data.count()}")
    return k_anonymous_data

#Step 6: L-diversity
def apply_l_diversity(data_df, quasi_identifiers, sensitive_attribute, l_value):
    print(f" L-diversity with l={l_value} on '{sensitive_attribute}'")
    l_diverse_data = data_df.groupBy(quasi_identifiers)\
        .agg(F.collect_set(sensitive_attribute).alias("sensitive_values_set"))\
        .filter(F.size("sensitive_values_set") >= l_value)\
        .join(data_df, on = quasi_identifiers, how = "inner")\
        .drop("sensitive_values_set")

    print(f" L-diversity row: {l_diverse_data.count()}")
    return l_diverse_data

#Step 7: Secure training simulation
def secure_training_simulation(df):
    X_tensor = torch.tensor(df[["Age"]].values, dtype=torch.float32)
    y_tensor = torch.tensor(df["Income"].values, dtype=torch.float32).view(-1, 1)

    half = len(y_tensor) // 2
    X1, X2 = X_tensor[:half], X_tensor[half:]
    y1, y2 = y_tensor[:half], y_tensor[half:]
    weights, bias = torch.rand((1, 1), requires_grad=True), torch.rand((1), requires_grad=True)

    overall_loss = ( ((X1 @ weights + bias - y1) ** 2).mean() + ((X2 @ weights + bias - y2) ** 2).mean() ) / 2

    print(f" Simulated Secure Training Loss: {overall_loss.item():.4f}")
    return overall_loss

#Step 8: Re-identification risk estimation
def reidentification_risk(original_df, transformed_df, quasi_identifiers):
    df_copy = original_df.copy()
    df_copy.columns = [col.lower().replace(" ", "_") for col in df_copy.columns]
    qi_normalized = [col.lower().replace(" ", "_") for col in quasi_identifiers]

    orig = df_copy[qi_normalized].dropna()
    trans = transformed_df.select([F.col(c).alias(c.lower().replace(" ", "_")) for c in quasi_identifiers]).dropna().drop_duplicates()
    trans_pd = trans.toPandas()

    merged = orig.merge(trans_pd, on=qi_normalized, how='inner').drop_duplicates()
    reid_rate = len(merged) / len(trans_pd) if len(trans_pd) > 0 else 0
    print(f" Re-identification Risk: {reid_rate:.4f} ({len(merged)} matched out of {len(trans_pd)})")
    return reid_rate

#Step 10: Secure training simulation on anonymized data
def secure_training_simulation_anonymized(df):
    X_tensor = torch.tensor(df[["age"]].values, dtype=torch.float32)
    y_tensor = torch.tensor(df["income"].values, dtype=torch.float32).view(-1, 1)

    half = len(y_tensor) // 2
    X1, X2 = X_tensor[:half], X_tensor[half:]
    y1, y2 = y_tensor[:half], y_tensor[half:]

    weights, bias = torch.rand((1, 1), requires_grad=True), torch.rand((1), requires_grad=True)
    overall_loss = ( ((X1 @ weights + bias - y1) ** 2).mean() + ((X2 @ weights + bias - y2) ** 2).mean() ) / 2

    print(f" Simulated Secure Training Loss from Anonymized Data: {overall_loss.item():.4f}")
    return overall_loss


#Step 11: Main function
def main():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    df = load_data(url)
    df = preprocess_data(df)

    # Setup simulated SMPC
    client_alice, client_bob = setup_smpc()

    # SMPC - Secure average on original data
    income_tensor = torch.tensor(df["Income"].values, dtype=torch.float32)
    secure_average(income_tensor)

    # Spark init
    spark = SparkSession.builder.appName("PrivacyPipeline").getOrCreate()

    # Define Spark schema
    ADULT_SCHEMA = StructType([
        StructField("age", IntegerType(), True),
        StructField("workclass", StringType(), True),
        StructField("fnlwgt", IntegerType(), True),
        StructField("education", StringType(), True),
        StructField("education_num", IntegerType(), True),
        StructField("marital_status", StringType(), True),
        StructField("occupation", StringType(), True),
        StructField("relationship", StringType(), True),
        StructField("race", StringType(), True),
        StructField("sex", StringType(), True),
        StructField("capital_gain", IntegerType(), True),
        StructField("capital_loss", IntegerType(), True),
        StructField("hours_per_week", IntegerType(), True),
        StructField("native_country", StringType(), True),
        StructField("income", StringType(), True),
    ])

    spark_df = spark.createDataFrame(df, schema=ADULT_SCHEMA)

    # K-anonymity + L-diversity
    QUASI_IDENTIFIERS = ["age", "workclass"]
    SENSITIVE_ATTRIBUTE = "income"
    k_anonymous_data = apply_k_anonymity(spark_df, QUASI_IDENTIFIERS, k_value=5)
    l_diverse_data = apply_l_diversity(k_anonymous_data, QUASI_IDENTIFIERS, SENSITIVE_ATTRIBUTE, l_value=2)

    # Convert income column in Spark DataFrame to numeric
    l_diverse_data_numeric_income = l_diverse_data.withColumn("income_numeric", F.when(F.col("income") == ">50K", 1.0).otherwise(0.0)).drop("income").withColumnRenamed("income_numeric", "income")

    # Check columns before training simulation
    print("Columns in L-diversity data with numeric income:")
    print(l_diverse_data_numeric_income.columns)

    # Re-identification risk
    reidentification_risk(df, l_diverse_data_numeric_income, QUASI_IDENTIFIERS)

    # SMPC on anonymized data (using the dataframe with numeric income)
    income_tensor_anonymized = torch.tensor(l_diverse_data_numeric_income.select("income").rdd.flatMap(lambda x: x).collect(), dtype=torch.float32)

    spark.stop()

if __name__ == "__main__":
    main()