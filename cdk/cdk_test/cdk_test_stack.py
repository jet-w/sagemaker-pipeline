from aws_cdk import (
    Stack,
    aws_ecr as ecr,
    aws_ecr_assets as assets,
    CfnOutput,
    RemovalPolicy,
    aws_sagemaker,
)
from constructs import Construct

class DockerBuildStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        #aws_sagemaker.DockerImageAsset()
        # Create an ECR repository
        #repository = ecr.Repository(
        #    self, 
        #    'mlops',
        #    repository_name='mlops',
        #    removal_policy=RemovalPolicy.DESTROY,  # NOT recommended for production
        #    auto_delete_images=True  # NOT recommended for production
        #)

        
        # Build Docker image from local Dockerfile
        docker_image = assets.DockerImageAsset(
            self, 
            'DockerImage',
            directory='../src/',  # Directory containing your Dockerfile
            file='steps/preprocess/Dockerfile',     # Name of the Dockerfile
            #platform=assets.Platform.LINUX_ARM64,
            # Optional: Build args if needed
            #build_args={
            #    # "ARG_NAME": "value"
            #}
        )
        print(docker_image.image_uri)
        # Output the image URI
        CfnOutput(
            self, 
            'ImageUri',
            value=docker_image.image_uri,
            description='Docker image URI in ECR'
        )