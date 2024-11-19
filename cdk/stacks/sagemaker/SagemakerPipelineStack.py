from aws_cdk import (
    Stack,
    aws_ecr_assets as ecr_assets,
    aws_sagemaker as sagemaker,
    aws_iam as iam,
    aws_s3 as s3,
    RemovalPolicy,
    CfnOutput
)
from constructs import Construct

class SageMakerPipelineStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Create S3 bucket for data and artifacts
        bucket = s3.Bucket(
            self, 
            'PipelineBucket',
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True
        )

        # Build Docker images for processing and training
        processing_image = ecr_assets.DockerImageAsset(
            self,
            'ProcessingImage',
            directory='../src/',
            file='steps/preprocess/Dockerfile'
        )

        training_image = ecr_assets.DockerImageAsset(
            self,
            'TrainingImage',
            directory='../src/steps/preprocess',
            file='Dockerfile'
        )

        # Create IAM role for SageMaker
        sagemaker_role = iam.Role(
            self,
            'SageMakerRole',
            assumed_by=iam.ServicePrincipal('sagemaker.amazonaws.com'),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name('AmazonSageMakerFullAccess'),
                iam.ManagedPolicy.from_aws_managed_policy_name('AmazonS3FullAccess')
            ]
        )

        # Create SageMaker Pipeline
        pipeline_name = 'CustomMLPipeline'
        
        # Create processing job definition
        processing_step = {
            "Name": "ProcessingStep",
            "ProcessingInputs": [{
                "InputName": "input-1",
                "S3Input": {
                    "S3Uri": f"s3://{bucket.bucket_name}/input",
                    "LocalPath": "/opt/ml/processing/input",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File"
                }
            }],
            "ProcessingOutputConfig": {
                "Outputs": [{
                    "OutputName": "processed-data",
                    "S3Output": {
                        "S3Uri": f"s3://{bucket.bucket_name}/processed",
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob"
                    }
                }]
            },
            "ProcessingResources": {
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": "ml.m5.xlarge",
                    "VolumeSizeInGB": 30
                }
            },
            "AppSpecification": {
                "ImageUri": processing_image.image_uri,
                "ContainerEntrypoint": ["python3", "/opt/ml/processing/input/process.py"]
            },
            "RoleArn": sagemaker_role.role_arn
        }

        # Create training job definition
        training_step = {
            "Name": "TrainingStep",
            "HyperParameters": {
                "epochs": "10",
                "batch-size": "32",
                "learning-rate": "0.001"
            },
            "InputDataConfig": [{
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3Uri": f"s3://{bucket.bucket_name}/processed",
                        "S3DataType": "S3Prefix",
                        "S3DataDistributionType": "FullyReplicated"
                    }
                }
            }],
            "OutputDataConfig": {
                "S3OutputPath": f"s3://{bucket.bucket_name}/model"
            },
            "ResourceConfig": {
                "InstanceCount": 1,
                "InstanceType": "ml.m5.xlarge",
                "VolumeSizeInGB": 30
            },
            "AlgorithmSpecification": {
                "TrainingImage": training_image.image_uri,
                "TrainingInputMode": "File"
            },
            "RoleArn": sagemaker_role.role_arn
        }

        # Create the pipeline
        pipeline = sagemaker.CfnPipeline(
            self,
            'MLPipeline',
            pipeline_name=pipeline_name,
            pipeline_definition={
                "Version": "2020-12-01",
                "PipelineDefinitionBody": {
                    "Steps": [
                        {
                            "ProcessingStep": processing_step
                        },
                        {
                            "TrainingStep": training_step
                        }
                    ]
                }
            },
            role_arn=sagemaker_role.role_arn
        )

        # Outputs
        CfnOutput(
            self,
            'PipelineName',
            value=pipeline_name,
            description='SageMaker Pipeline Name'
        )
        
        CfnOutput(
            self,
            'BucketName',
            value=bucket.bucket_name,
            description='S3 Bucket Name'
        )