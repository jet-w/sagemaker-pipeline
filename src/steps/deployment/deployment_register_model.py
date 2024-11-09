import argparse
import boto3
import logging
import sagemaker
from botocore.exceptions import ClientError
import time

from sagemaker import ModelPackage

logger = logging.getLogger(__name__)
sm_client = boto3.client("sagemaker")

def get_approved_package(model_package_group_name):
    try:
        # Get the latest approved model package
        response = sm_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            MaxResults=100,
        )
        approved_packages = response["ModelPackageSummaryList"]

        # Fetch more packages if none returned with continuation token
        while len(approved_packages) == 0 and "NextToken" in response:
            logger.debug("Getting more packages for token: {}".format(response["NextToken"]))
            response = sm_client.list_model_packages(
                ModelPackageGroupName=model_package_group_name,
                ModelApprovalStatus="Approved",
                SortBy="CreationTime",
                MaxResults=100,
                NextToken=response["NextToken"],
            )
            approved_packages.extend(response["ModelPackageSummaryList"])

        # Return error if no packages found
        if len(approved_packages) == 0:
            error_message = (
                f"No approved ModelPackage found for ModelPackageGroup: {model_package_group_name}"
            )
            logger.error(error_message)
            raise Exception(error_message)

        # Return the pmodel package arn
        model_package_arn = approved_packages[0]["ModelPackageArn"]
        logger.info(f"Identified the latest approved model package: {model_package_arn}")
        return approved_packages[0]
        # return model_package_arn
    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        logger.error(error_message)
        raise Exception(error_message)


model_package_group_name = "HumanSystemsAIOpsModelPackageGroup"
role = "SageMaker-ExecutionRole-20241030T121452"

if __name__ == "__main__":
    sess = boto3.Session()
    sm_client = sess.client("sagemaker")
    #sm = sess.client("sagemaker")
    sagemaker_session = sagemaker.Session(boto_session=sess)

    pck = get_approved_package(
        model_package_group_name
    )  
    # Reminder: model_package_group_name was defined as "NominetAbaloneModelPackageGroupName" at the beginning of the pipeline definition
    model_description = sm_client.describe_model_package(ModelPackageName=pck["ModelPackageArn"])
    model_package_arn = model_description["ModelPackageArn"]
    model = ModelPackage(
        role=role, model_package_arn=model_package_arn, sagemaker_session=sagemaker_session
    )
    
    endpoint_name = "HS-endpoint-" + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
    print("EndpointName= {}".format(endpoint_name))
    model.deploy(
        initial_instance_count=1, 
        instance_type="ml.t2.medium", 
        endpoint_name=endpoint_name
    )