import boto3
import os
from aws_cdk import aws_ecr_assets

ecr_client = boto3.client('ecr')


def get_img_uri(docker_path, docker_file):
    return aws_ecr_assets.DockerImageAsset(
        scope,
        "ApiPrivateImageAsset",
        directory=os.path.join(docker_path, docker_file),
        exclude=["web_testlab"],
        file="api_private/Dockerfile"
    ).image_uri
