
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from etc import input_data, role, processing_instance_count, bucket
import logging
import time
from .utils import get_approved_package
import boto3




sm_client = boto3.client("sagemaker")

pck = get_approved_package(
    model_package_group_name
)  # Reminder: model_package_group_name was defined as "NominetAbaloneModelPackageGroupName" at the beginning of the pipeline definition
model_description = sm_client.describe_model_package(ModelPackageName=pck["ModelPackageArn"])

from sagemaker import ModelPackage

model_package_arn = model_description["ModelPackageArn"]
model = ModelPackage(
    role=role, model_package_arn=model_package_arn, sagemaker_session=sagemaker_session
)

endpoint_name = "DEMO-endpoint-" + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
print("EndpointName= {}".format(endpoint_name))
model.deploy(initial_instance_count=1, instance_type="ml.t3.medium", endpoint_name=endpoint_name)

#def get_deployment_args(pipeline_session):
#    s3_train  = f"s3://{bucket}/humansystem/preprocess/output/train"
#    s3_test   = f"s3://{bucket}/humansystem/preprocess/output/test"
#
#    sklearn_framework_version = "1.2-1"
#    sklearn_processor = SKLearnProcessor(
#        framework_version=sklearn_framework_version,
#        instance_type="ml.t3.medium",
#        instance_count=processing_instance_count,
#        base_job_name="sklearn-housing-data-process",
#        role=role,
#        sagemaker_session=pipeline_session,
#        #image_uri=""
#    )
#    
#    #processor_args = sklearn_processor.run(
#    return sklearn_processor.run(
#        inputs=[
#            ProcessingInput(source=input_data, destination="/opt/ml/processing/input"),
#        ],
#        outputs=[
#            ProcessingOutput(output_name="train", source="/opt/ml/processing/train", destination=s3_train),
#            ProcessingOutput(output_name="test", source="/opt/ml/processing/test", destination=s3_test),
#        ],
#        code="steps/preprocess/preprocess.py",
#    )

