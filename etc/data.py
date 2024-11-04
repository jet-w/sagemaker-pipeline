import os
import sagemaker
import boto3
import pandas as pd

def get_execution_role():
    return "SageMaker-ExecutionRole-20241030T121452"

sess              = boto3.Session()
sm                = sess.client("sagemaker")
role              = get_execution_role()
role_arn          = f"arn:aws:iam::654654179472:role/service-role/{role}"
sagemaker_session = sagemaker.Session(boto_session=sess)
bucket            = "shared-hs-mlops-bucket" #sagemaker_session.default_bucket()
region            = boto3.Session().region_name


model_package_group_name = "PipelineModelPackageGroup"
prefix                   = "mas-pipeline-model-example"
pipeline_name            = "mas-serial-inference-pipeline"  # SageMaker Pipeline name


# Where to store the trained model
model_path = f"s3://{bucket}/{prefix}/model/"

raw_dir = os.path.join(os.getcwd(), "data", "raw")

################################################################################
raw_s3 = "s3://shared-hs-mlops-bucket/mas-pipeline-model-example/data/raw"
s3 = boto3.client("s3")
s3.download_file(
    f"sagemaker-example-files-prod-{region}",
    "datasets/tabular/california_housing/cal_housing.tgz",
    "cal_housing.tgz",
)

feature_columns = [
    "longitude",
    "latitude",
    "housingMedianAge",
    "totalRooms",
    "totalBedrooms",
    "population",
    "households",
    "medianIncome",
]
label_column    = "medianHouseValue"
columns = feature_columns + label_column

cal_housing_df = pd.read_csv("CaliforniaHousing/cal_housing.data", names=columns, header=None)
# Scaling target down to avoid overcomplicating the example
cal_housing_df["medianHouseValue"] /= 500000  
cal_housing_df.to_csv(f"./data/raw/raw_data_all.csv", header=True, index=False)
rawdata_s3_prefix = "{}/data/raw".format(prefix)
data_path = os.path.join(os.path.dirname(__file__), "../data/raw/")
raw_s3 = sagemaker_session.upload_data(bucket = bucket, path=data_path, key_prefix=rawdata_s3_prefix)


