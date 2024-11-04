import os
from sagemaker.workflow.parameters import ParameterInteger, ParameterString, ParameterFloat
from sagemaker.workflow.pipeline_context import PipelineSession, LocalPipelineSession
import sagemaker
import boto3

tensorflow_version = "2.4.1"
sklearn_framework_version = "1.2-1"

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

raw_dir = os.path.join(os.getcwd(), "data", "raw")
os.makedirs(raw_dir, exist_ok=True)
#data_dir = os.path.join(os.getcwd(), "data")
#os.makedirs(data_dir, exist_ok=True)
#raw_dir = os.path.join(os.getcwd(), "data/raw")
#os.makedirs(raw_dir, exist_ok=True)

raw_s3 = "s3://shared-hs-mlops-bucket/mas-pipeline-model-example/data/raw"

#pipeline_session = PipelineSession()
pipeline_session = LocalPipelineSession()

# raw input data
input_data               = ParameterString(name="InputData", default_value=raw_s3)
# status of newly trained model in registry
model_approval_status    = ParameterString(name="ModelApprovalStatus", default_value="Approved")
# processing step parameters
processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.m5.xlarge")
processing_instance_count= ParameterInteger(name="ProcessingInstanceCount", default_value=1)

# training step parameters
training_instance_type   = ParameterString(name="TrainingInstanceType", default_value="ml.m5.xlarge")
training_epochs          = ParameterString(name="TrainingEpochs", default_value="100")

# model performance step parameters
accuracy_mse_threshold   = ParameterFloat(name="AccuracyMseThreshold", default_value=0.75)
