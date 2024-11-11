import os
import argparse
import subprocess
import sys

def install_package(package_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

install_package("boto3==1.28.57")
install_package("sagemaker")

import sagemaker
from sagemaker import ModelPackage
# deploy_model.py
import boto3
sm_client = boto3.client("sagemaker", region_name='us-east-1')

def deploy_model(model_s3_uri, endpoint_name, instance_type, role_arn):
    print("model_s3_uri:", model_s3_uri)
    print("endpoint_name", endpoint_name)
    print("instance_type", instance_type)
    print("role_arn", role_arn)

    # Register the model (optional: skip if model is already registered)
    model_name = endpoint_name + "-model"
    model_data_url = model_s3_uri

    container_def = {
        "Image": "<your-image-uri>",  # specify your image URI
        "ModelDataUrl": model_data_url,
        "Environment": {
            "SAGEMAKER_PROGRAM": "inference.py",
            "SAGEMAKER_SUBMIT_DIRECTORY": model_data_url
        }
    }

    sm_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role_arn,
        PrimaryContainer=container_def
    )

    # Deploy the model to a SageMaker endpoint
    sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_name + "-config",
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InstanceType": instance_type,
                "InitialInstanceCount": 1,
            },
        ],
    )

    sm_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_name + "-config"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-s3-uri", type=str)
    parser.add_argument("--model-package-arn", type=str)
    parser.add_argument("--endpoint-name", type=str)
    parser.add_argument("--instance-type", type=str)
    parser.add_argument("--role-arn", type=str)
    
    return parser.parse_args()


def deploy_model_pkg_arn(model_pkg_arn, endpoint_name, instance_type):
    sess              = boto3.Session()
    sagemaker_session = sagemaker.Session(boto_session=sess)
    
    model_package_arn = model_pkg_arn
    model = ModelPackage(
        role=sagemaker.get_execution_role(), 
        model_package_arn=model_package_arn, 
        sagemaker_session=sagemaker_session
    )
    predictor = model.deploy(
        initial_instance_count=1, 
        #instance_type="ml.m5.large",
        instance_type=instance_type,
        # [
        #    ml.t2.medium, ml.t2.large, ml.t2.xlarge, ml.t2.2xlarge, 
        #    ml.m4.xlarge, ml.m4.2xlarge, ml.m4.4xlarge, ml.m4.10xlarge, ml.m4.16xlarge, 
        #    ml.m5.large, ml.m5.xlarge, ml.m5.2xlarge, ml.m5.4xlarge, ml.m5.12xlarge, ml.m5.24xlarge, 
        #    ml.m5d.large, ml.m5d.xlarge, ml.m5d.2xlarge, ml.m5d.4xlarge, ml.m5d.12xlarge, ml.m5d.24xlarge, 
        #    ml.c4.large, ml.c4.xlarge, ml.c4.2xlarge, ml.c4.4xlarge, ml.c4.8xlarge, 
        #    ml.p2.xlarge, ml.p2.8xlarge, ml.p2.16xlarge, 
        #    ml.p3.2xlarge, ml.p3.8xlarge, ml.p3.16xlarge, 
        #    ml.c5.large, ml.c5.xlarge, ml.c5.2xlarge, ml.c5.4xlarge, ml.c5.9xlarge, ml.c5.18xlarge, 
        #    ml.c5d.large, ml.c5d.xlarge, ml.c5d.2xlarge, ml.c5d.4xlarge, ml.c5d.9xlarge, ml.c5d.18xlarge, 
        #    ml.g4dn.xlarge, ml.g4dn.2xlarge, ml.g4dn.4xlarge, ml.g4dn.8xlarge, ml.g4dn.12xlarge, ml.g4dn.16xlarge, 
        #    ml.r5.large, ml.r5.xlarge, ml.r5.2xlarge, ml.r5.4xlarge, ml.r5.12xlarge, ml.r5.24xlarge, 
        #    ml.r5d.large, ml.r5d.xlarge, ml.r5d.2xlarge, ml.r5d.4xlarge, ml.r5d.12xlarge, ml.r5d.24xlarge, 
        #    ml.inf1.xlarge, ml.inf1.2xlarge, ml.inf1.6xlarge, ml.inf1.24xlarge, 
        #    ml.dl1.24xlarge, 
        #    ml.c6i.large, ml.c6i.xlarge, ml.c6i.2xlarge, ml.c6i.4xlarge, ml.c6i.8xlarge, ml.c6i.12xlarge, ml.c6i.16xlarge, ml.c6i.24xlarge, ml.c6i.32xlarge, 
        #    ml.g5.xlarge, ml.g5.2xlarge, ml.g5.4xlarge, ml.g5.8xlarge, ml.g5.12xlarge, ml.g5.16xlarge, ml.g5.24xlarge, ml.g5.48xlarge, 
        #    ml.g6.xlarge, ml.g6.2xlarge, ml.g6.4xlarge, ml.g6.8xlarge, ml.g6.12xlarge, ml.g6.16xlarge, ml.g6.24xlarge, ml.g6.48xlarge, 
        #    ml.p4d.24xlarge, 
        #    ml.c7g.large, ml.c7g.xlarge, ml.c7g.2xlarge, ml.c7g.4xlarge, ml.c7g.8xlarge, ml.c7g.12xlarge, ml.c7g.16xlarge, 
        #    ml.m6g.large, ml.m6g.xlarge, ml.m6g.2xlarge, ml.m6g.4xlarge, ml.m6g.8xlarge, ml.m6g.12xlarge, ml.m6g.16xlarge, 
        #    ml.m6gd.large, ml.m6gd.xlarge, ml.m6gd.2xlarge, ml.m6gd.4xlarge, ml.m6gd.8xlarge, ml.m6gd.12xlarge, ml.m6gd.16xlarge, 
        #    ml.c6g.large, ml.c6g.xlarge, ml.c6g.2xlarge, ml.c6g.4xlarge, ml.c6g.8xlarge, ml.c6g.12xlarge, ml.c6g.16xlarge, ml.c6gd.large, 
        #    ml.c6gd.xlarge, ml.c6gd.2xlarge, ml.c6gd.4xlarge, ml.c6gd.8xlarge, ml.c6gd.12xlarge, ml.c6gd.16xlarge, 
        #    ml.c6gn.large, ml.c6gn.xlarge, ml.c6gn.2xlarge, ml.c6gn.4xlarge, ml.c6gn.8xlarge, ml.c6gn.12xlarge, ml.c6gn.16xlarge, 
        #    ml.r6g.large, ml.r6g.xlarge, ml.r6g.2xlarge, ml.r6g.4xlarge, ml.r6g.8xlarge, ml.r6g.12xlarge, ml.r6g.16xlarge, 
        #    ml.r6gd.large, ml.r6gd.xlarge, ml.r6gd.2xlarge, ml.r6gd.4xlarge, ml.r6gd.8xlarge, ml.r6gd.12xlarge, ml.r6gd.16xlarge, 
        #    ml.p4de.24xlarge, ml.trn1.2xlarge, ml.trn1.32xlarge, 
        #    ml.trn1n.32xlarge, 
        #    ml.inf2.xlarge, ml.inf2.8xlarge, ml.inf2.24xlarge, ml.inf2.48xlarge, ml.inf2e.32xlarge, ml.p5.48xlarge, ml.m7i.large, ml.m7i.xlarge, ml.m7i.2xlarge, ml.m7i.4xlarge, ml.m7i.8xlarge, ml.m7i.12xlarge, ml.m7i.16xlarge, ml.m7i.24xlarge, ml.m7i.48xlarge, ml.c7i.large, ml.c7i.xlarge, ml.c7i.2xlarge, ml.c7i.4xlarge, ml.c7i.8xlarge, ml.c7i.12xlarge, ml.c7i.16xlarge, ml.c7i.24xlarge, ml.c7i.48xlarge, ml.r7i.large, ml.r7i.xlarge, ml.r7i.2xlarge, ml.r7i.4xlarge, ml.r7i.8xlarge, ml.r7i.12xlarge, ml.r7i.16xlarge, ml.r7i.24xlarge, ml.r7i.48xlarge]
        endpoint_name=endpoint_name
    )
    pass

if __name__ == "__main__":
    args = parse_args()
    
    model_s3_uri =args.model_s3_uri if hasattr(args, "model_s3_uri") else None
    model_package_arn =args.model_package_arn if hasattr(args, "model_package_arn") else None

    if model_s3_uri is not None:
        deploy_model(
            model_s3_uri=args.model_s3_uri,
            endpoint_name=args.endpoint_name,
            instance_type=args.instance_type,
            role_arn=args.role_arn
        )
    elif model_package_arn is not None:
        deploy_model_pkg_arn(
            model_pkg_arn=args.model_package_arn,
            endpoint_name=args.endpoint_name,
            instance_type=args.instance_type,
        )
    #deploy_model_pkg_arn(
    #    model_pkg_arn=args.model_package_arn,
    #    endpoint_name=args.endpoint_name,
    #    instance_type=args.instance_type
    #)
