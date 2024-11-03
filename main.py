import os
import time
import boto3
import numpy as np
import pandas as pd
import sagemaker
#from sagemaker import get_execution_role

def get_execution_role():
    return "SageMaker-ExecutionRole-20241030T121452"

sess              = boto3.Session()
sm                = sess.client("sagemaker")
role              = get_execution_role()
sagemaker_session = sagemaker.Session(boto_session=sess)
bucket            = "shared-hs-mlops-bucket" #sagemaker_session.default_bucket()
region            = boto3.Session().region_name


model_package_group_name = "PipelineModelPackageGroup"
prefix                   = "mas-pipeline-model-example"
pipeline_name            = "mas-serial-inference-pipeline"  # SageMaker Pipeline name

raw_dir = os.path.join(os.getcwd(), "data", "raw")

#data_dir = os.path.join(os.getcwd(), "data")
#os.makedirs(data_dir, exist_ok=True)
#raw_dir = os.path.join(os.getcwd(), "data/raw")
#os.makedirs(raw_dir, exist_ok=True)

raw_s3 = "s3://shared-hs-mlops-bucket/mas-pipeline-model-example/data/raw"


