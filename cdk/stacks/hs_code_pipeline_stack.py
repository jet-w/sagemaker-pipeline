from aws_cdk import (
    RemovalPolicy,
    Stack,
    aws_codebuild,
    aws_codecommit,
    aws_s3,
    pipelines,
)
from constructs import Construct

from typing import Literal

Environment = Literal["dev", "stage", "prod"]

def resource_name(name: str, branch: Environment) -> str:
    """Helper function to consistently name resources"""
    return f"hs-mlops-{name}-{branch}"


class Pipeline(Stack):
    """
    This pipeline is deployed in the Deployment account and is responsible for deploying resources
    for the given branch (dev / stage / prod).
    """
    repository_name = ""
    def __init__(
        self, scope: Construct, id: str, branch: Environment, **kwargs
    ) -> None:
        super().__init__(scope, id, **kwargs)

        # S3 bucket to store the CodePipeline build artifacts
        artifact_bucket_name = resource_name("pipeline-artifacts", branch)
        artifact_bucket = aws_s3.Bucket(
            self,
            artifact_bucket_name,
            bucket_name=artifact_bucket_name,
            block_public_access=aws_s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            encryption=aws_s3.BucketEncryption.KMS,
        )

        # CodeCommit source
        source = pipelines.CodePipelineSource.code_commit(
            repository=aws_codecommit.Repository.from_repository_name(
                self,
                resource_name("repository", branch),
                repository_name="hs-mlops",
            ),
            branch=branch,
        )

        # Steps to install Poetry and set up the environment in a CodeBuild container
        install_commands = [
            "curl -sSL https://install.python-poetry.org | python3 -",
            'export PATH="/root/.local/bin:$PATH"',
            "python --version",
            "poetry --version",
            "pwd"
        ]

        # Create a self-mutating CDK pipeline within the deployment AWS account
        pipeline_name = resource_name("pipeline", branch)
        pipeline = pipelines.CodePipeline(
            self,
            pipeline_name,
            pipeline_name=pipeline_name,
            artifact_bucket=artifact_bucket,
            synth=pipelines.ShellStep(
                "Synth",
                install_commands=install_commands,
                input=source,
                # We pass in a BUILD_ENV environment variable ('dev', 'stage' or 'prod') because:
                # 1. We have to build the TestLab Vue static website as part of this synth step
                #       - The built files in '/dist' are then deployed to S3 as part of the later CDK deploy wave
                # 2. The TestLab Vue app needs to know which environment it's being deployed to during build time
                #       - We run 'npm run build-dev' or 'npm run build-stage' etc. which picks up the right '.env.{BRANCH}' config files
                # TODO: The Vue app probably shouldn't care what environment it's being deployed to at build time.
                commands=[f"BUILD_ENV={branch} make synth"],
                primary_output_directory="cdk/cdk.out",
            ),
        )

        # Add step to run our tests
        pipeline.add_wave(
            "Test",
            post=[
                pipelines.CodeBuildStep(
                    resource_name("lint-and-test", branch),
                    install_commands=install_commands,
                    input=source,
                    # Run the linting tasks sequentially with --jobs=1, because CodeBuild is failing
                    # when running them in parallel :(
                    commands=["make install && make lint --jobs=1 && make test"],
                    # We want to spin up a local dynamodb instance with docker for testing
                    # so we'll run the tests in a privileged container
                    build_environment=aws_codebuild.BuildEnvironment(privileged=True),
                )
            ],
        )

        # Deploy our infrastructure
        #deploy_wave = pipeline.add_wave("Deploy")
        #deploy_wave.add_stage(DeployStage(self, "Deploy", branch=branch))
