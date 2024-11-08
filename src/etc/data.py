import os
import sagemaker
import boto3
import pandas as pd

def get_execution_role():
    return "SageMaker-ExecutionRole-20241030T121452"

sess              = boto3.Session()
sm                = sess.client("sagemaker")
#role              = sagemaker.get_execution_role() # get_execution_role()
role              = get_execution_role()
role_arn          = f"arn:aws:iam::654654179472:role/service-role/{role}"

sagemaker_session = sagemaker.Session(boto_session=sess)

bucket            = "shared-hs-mlops-bucket"             #sagemaker_session.default_bucket()
region            = sess.region_name


model_package_group_name = "HumanSystemsAIOpsModelPackageGroup"
prefix                   = "humansystems"
pipeline_name            = "humansystem-aiops-pipeline"  # SageMaker Pipeline name



raw_dir = os.path.join(os.getcwd(), "data", "raw")

################################################################################
raw_s3 = "s3://shared-hs-mlops-bucket/humansystem/preprocess/input/"
s3_model_output = "s3://shared-hs-mlops-bucket/humansystem/preprocess/output/"

label_column    = ["Peer-Work", "Reflection", "Additional-Resources", "Reminders"]