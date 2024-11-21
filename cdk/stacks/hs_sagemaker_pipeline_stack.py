from aws_cdk import (
    CfnOutput,
    aws_kms as kms,
    RemovalPolicy,
    Stack,
    aws_s3 as s3,
    aws_iam as iam,
    aws_codecommit as cc,
    aws_codebuild as cb,
    aws_iam as iam,
    Duration,
    aws_codepipeline as cp,
    aws_lambda as _lambda,
    aws_codepipeline_actions as cpactions,
    aws_s3 as s3,
    RemovalPolicy,
    aws_codecommit as cc,
)
import json
from cdk_nag import NagSuppressions
from constructs import Construct

class HSSagemakerPipelineStack(Stack):
    # Standard definition for CDK stack
    def __init__(self, scope: Construct, construct_id: str,  **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        #stack_name = Stack.of(self).stack_name.lower()
        
        filename = "project_config.json"
        variables = json.load(open(filename))
        
        sm_pipeline_name         = variables["SageMaker"]["Pipeline"]["Name"]
        sm_pipeline_role         = variables["SageMaker"]["Pipeline"]["IAMRole"]["Name"]
        sm_repo_name             = variables["CodeCommit"]["Name"]
        sm_code_build_s3_name    = variables["S3"]["CodeBuild"]["Name"]
        sm_code_artifact_s3_name = variables["S3"]["CodeBuild"]["Name"]
        sm_code_pipeline_name    = variables["CodePipeline"]["Name"]
        USE_AMT = variables["USE_AMT"]

        ## S3 bucket for access logs
        access_logs_bucket = s3.Bucket(
            self,
            "AccessLogsBucket",
            removal_policy= RemovalPolicy.DESTROY,
            block_public_access= s3.BlockPublicAccess.BLOCK_ALL,
            encryption= s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            auto_delete_objects=True,
        )

        pipeline_project_role = iam.Role(
                                self,
                                "Project Role",
                                role_name=sm_pipeline_role,
                                assumed_by=iam.CompositePrincipal(
                                    iam.ServicePrincipal("codebuild.amazonaws.com"),
                                    iam.ServicePrincipal("sagemaker.amazonaws.com"),
                                ),  
                                managed_policies=[
                                    iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"),#When you are moving to production follow least privilage access. Use custom policies instead of managed access policies
                                ]
                            )
        
        repository = cc.Repository(
                self,
                "Regression Pipeline Code",
                repository_name=sm_repo_name,
                #repository_name=f"sm-pipeline-regression-code",
                description="SageMaker Model building workflow infrastructure as code for the Project",
                code=cc.Code.from_directory('../src/'),
            )
        
        code_build_proj_s3  = s3.Bucket(
            self,
            "Pipeline Bucket",
            bucket_name=sm_code_build_s3_name,
            removal_policy= RemovalPolicy.DESTROY,
            block_public_access= s3.BlockPublicAccess.BLOCK_ALL,
            encryption= s3.BucketEncryption.S3_MANAGED,
            server_access_logs_bucket=access_logs_bucket,
            enforce_ssl=True,
            auto_delete_objects=True,
        )
        code_build_proj_s3.grant_read_write(pipeline_project_role)
        repository.grant_read(pipeline_project_role)

        # CodePipeline artifact_bucket ecryption key
        pipeline_encryption_key = kms.Key(
            self,
            'PipelineEncryptionKey',
            alias = 'codepipeline/workload',
            description = 'Encryption key for workload codepipeline artifact_bucket',
            enabled = True,
            enable_key_rotation = True,
            removal_policy=RemovalPolicy.DESTROY,
        )

        cb_project = cb.PipelineProject(
            self,
            "CB Pipeline Project",
            project_name=f"{sm_pipeline_name}-modelbuild",
            description="Builds the model building workflow code repository, creates the SageMaker Pipeline",
            encryption_key= pipeline_encryption_key,
            role=pipeline_project_role,
            environment_variables={
                "SM_PIPELINE_NAME": cb.BuildEnvironmentVariable(value=sm_pipeline_name),
                "ARTIFACT_BUCKET": cb.BuildEnvironmentVariable(value=code_build_proj_s3.bucket_name),
                "SAGEMAKER_PIPELINE_ROLE_ARN": cb.BuildEnvironmentVariable(value=pipeline_project_role.role_arn),
                "USE_AMT": cb.BuildEnvironmentVariable(value=USE_AMT),
                #"SM_MODEL_DEPLOY_LAMBDA_ARN": cb.BuildEnvironmentVariable(value=sm_model_deploy_lambda.function_arn),
            },
            environment=cb.BuildEnvironment(
                build_image=cb.LinuxBuildImage.STANDARD_7_0, #Use latest one!! More info: https://docs.aws.amazon.com/codebuild/latest/userguide/build-env-ref-available.html
                compute_type=cb.ComputeType.LARGE,           #If you need bigger compute, please change it to SMALL, MEDIUM, LARGE, X2_LARGE
            ),
            build_spec=cb.BuildSpec.from_source_filename("buildspec.yml"),
            timeout=Duration.minutes(480),
        )

        sm_pipeline_src_artifact = cp.Artifact()

        artifact_bucket = s3.Bucket(self,
                                    "MlOpsArtifactsBucket",
                                    bucket_name=sm_code_artifact_s3_name,
                                    removal_policy= RemovalPolicy.DESTROY,
                                    block_public_access= s3.BlockPublicAccess.BLOCK_ALL,
                                    encryption= s3.BucketEncryption.S3_MANAGED,
                                    server_access_logs_bucket=access_logs_bucket,
                                    enforce_ssl=True,
                                    auto_delete_objects=True,
                                )

        code_pipeline = cp.Pipeline(
                                    self,
                                    "Pipeline",
                                    artifact_bucket=artifact_bucket,
                                    #pipeline_name=f"{sm_pipeline_name}-modelbuild",
                                    pipeline_name=sm_code_pipeline_name,
                                    restart_execution_on_update=True,
                                    stages=[
                                        cp.StageProps(
                                            stage_name="Source",
                                            actions=[
                                                cpactions.CodeCommitSourceAction(
                                                    action_name="WorkflowCode",
                                                    repository=repository,
                                                    output=sm_pipeline_src_artifact,
                                                    branch="main",
                                                )
                                            ],
                                        ),
                                        cp.StageProps(
                                            stage_name="Build",
                                            actions=[
                                                cpactions.CodeBuildAction(
                                                    action_name="BuildAndCreateSageMakerPipeline",
                                                    input=sm_pipeline_src_artifact,
                                                    project=cb_project,
                                                )
                                            ],
                                        ),
                                    ],
                                )
        
        CfnOutput(self, "access_log_bucket_arn", export_name="accesslogbucketarn", value=access_logs_bucket.bucket_arn)
        CfnOutput(self, "pipeline_project_role_arn", export_name="pipelineprojectrolearn", value=pipeline_project_role.role_arn)
        
        ## CDK Nag Suppression
        NagSuppressions.add_resource_suppressions(pipeline_project_role,
                            suppressions=[{
                                            "id": "AwsSolutions-IAM5",
                                            "reason": "SageMaker Project provider requires access to all indicies",
                                            }
                                        ],
                            apply_to_children=True)
        
        ## CDK Nag Suppression
        NagSuppressions.add_resource_suppressions(code_pipeline,
                            suppressions=[{
                                            "id": "AwsSolutions-IAM5",
                                            "reason": "Code pipeline role requires access to s3 bucket all indicies",
                                            }
                                        ],
                            apply_to_children=True)
        
        ### CDK Nag Suppression
        #NagSuppressions.add_resource_suppressions([sm_model_deploy_lambda.role, pipeline_project_role],
        #                    suppressions=[{
        #                                    "id": "AwsSolutions-IAM4",
        #                                    "reason": "Allowing AmazonSageMakerFullAccess as it is sample code, for production usecase scope down the permission",
        #                                    }
        #                                ],
        #                    apply_to_children=True)
        ## CDK Nag Suppression
        NagSuppressions.add_stack_suppressions(self, [
                                {
                                    "id": 'AwsSolutions-IAM4',
                                    "reason": 'Lambda execution policy for custom resources created by higher level CDK constructs',
                                    "appliesTo": [
                                            'Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
                                        ],
                                }])